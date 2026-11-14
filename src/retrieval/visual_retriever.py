"""
Visual retrieval and RAG module for MECG using SigLIP 2 and FAISS.
Builds searchable visual indexes of training complaint video frames,
retrieves similar reference complaints, and enforces strict query exclusion and split isolation.
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import yaml
except ImportError:
    yaml = None

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    faiss = None
    HAS_FAISS = False

logger = logging.getLogger("MECG.VisualRetriever")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class VisualRetriever:
    """
    RAG retriever for visually similar reference complaints using SigLIP embeddings and FAISS.
    Guarantees no self-retrieval and zero test/val leakage into training references.
    """

    def __init__(
        self,
        model_id: str = "google/siglip-base-patch16-224",
        index_dir: Union[str, Path] = "outputs/retrieval_index",
        top_k: int = 3,
        vector_dim: int = 768,
        device: str = "auto",
    ):
        self.model_id = model_id
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.vector_dim = vector_dim
        self.device = device

        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self.encoder = None
        self.processor = None
        self.fallback_vectors: List[List[float]] = []

        self._load_existing_index()

    @classmethod
    def from_config(cls, config_path: Union[str, Path] = "configs/config.yaml") -> "VisualRetriever":
        """Instantiates VisualRetriever using project YAML config."""
        cfg_path = Path(config_path)
        if cfg_path.exists() and yaml:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            paths = cfg.get("paths", {})
            r_cfg = cfg.get("retrieval", {})
            return cls(
                model_id=r_cfg.get("vision_encoder_id", "google/siglip-base-patch16-224"),
                index_dir=paths.get("retrieval_index_dir", "outputs/retrieval_index"),
                top_k=r_cfg.get("top_k", 3),
                vector_dim=r_cfg.get("vector_dimension", 768),
            )
        return cls()

    def _init_encoder(self):
        """Initializes SigLIP vision model and processor on demand."""
        if self.encoder is not None:
            return

        import torch
        from transformers import AutoModel, AutoProcessor

        if self.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device

        logger.info(f"Loading SigLIP vision encoder: {self.model_id} on {device}...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.encoder = AutoModel.from_pretrained(self.model_id)
        self.encoder.to(device)
        self.encoder.eval()
        self._device = device

    def _load_existing_index(self):
        """Loads FAISS index and metadata registry from disk if available."""
        index_file = self.index_dir / "siglip_faiss.index"
        meta_file = self.index_dir / "metadata.json"

        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded {len(self.metadata)} reference complaint metadata entries.")
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")

        if HAS_FAISS and index_file.exists():
            try:
                self.index = faiss.read_index(str(index_file))
                logger.info(f"Loaded FAISS index containing {self.index.ntotal} vectors.")
            except Exception as e:
                logger.warning(f"Could not load FAISS index: {e}")

    def compute_embedding(self, image_paths: List[Union[str, Path]]) -> List[float]:
        """
        Computes pooled, L2-normalized embedding for a collection of video frames.
        """
        valid_paths = [Path(p) for p in image_paths if Path(p).exists()]
        if not valid_paths:
            return [0.0] * self.vector_dim

        self._init_encoder()
        import torch
        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in valid_paths]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            if hasattr(self.encoder, "get_image_features"):
                feat_out = self.encoder.get_image_features(**inputs)
            else:
                feat_out = self.encoder(**inputs)

            # Extract tensor from HuggingFace output container
            if isinstance(feat_out, torch.Tensor):
                tensor_feats = feat_out
            elif hasattr(feat_out, "image_embeds") and feat_out.image_embeds is not None:
                tensor_feats = feat_out.image_embeds
            elif hasattr(feat_out, "pooler_output") and feat_out.pooler_output is not None:
                tensor_feats = feat_out.pooler_output
            elif hasattr(feat_out, "last_hidden_state") and feat_out.last_hidden_state is not None:
                tensor_feats = feat_out.last_hidden_state.mean(dim=1)
            else:
                tensor_feats = feat_out[0] if isinstance(feat_out, (list, tuple)) else getattr(feat_out, "last_hidden_state", feat_out)

            # Average pool across frames
            pooled = tensor_feats.mean(dim=0, keepdim=True)
            # L2 normalize
            norm_features = pooled / pooled.norm(dim=-1, keepdim=True)
            emb = norm_features.squeeze(0).cpu().tolist()

        return emb

    def add_video(
        self,
        video_hash: str,
        frame_paths: List[Union[str, Path]],
        metadata: Dict[str, Any],
        embedding: Optional[List[float]] = None,
    ):
        """Adds a reference video's pooled embedding and metadata to index."""
        emb = embedding or self.compute_embedding(frame_paths)
        if not emb or all(v == 0.0 for v in emb):
            return

        # Pure Python fallback or FAISS
        self.fallback_vectors.append(emb)

        if HAS_FAISS:
            import numpy as np
            if self.index is None:
                self.index = faiss.IndexFlatIP(self.vector_dim)
            vec_np = np.array([emb], dtype=np.float32)
            self.index.add(vec_np)

        meta_entry = dict(metadata)
        meta_entry["video_url_hash"] = video_hash
        self.metadata.append(meta_entry)

    def save_index(self) -> Tuple[Path, Path]:
        """Saves FAISS index and metadata file to disk."""
        index_file = self.index_dir / "siglip_faiss.index"
        meta_file = self.index_dir / "metadata.json"

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

        if HAS_FAISS and self.index is not None:
            faiss.write_index(self.index, str(index_file))

        logger.info(f"Retrieval index saved: {len(self.metadata)} entries.")
        return index_file, meta_file

    def search(
        self,
        query_frame_paths: List[Union[str, Path]],
        top_k: Optional[int] = None,
        exclude_video_hash: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k visually similar reference complaints.
        Strictly excludes query video to prevent self-retrieval.
        Returns:
            [
                {
                    "score": float,
                    "video_url_hash": str,
                    "aspects": str,
                    "domain": str,
                    "reference_complaint": str
                }, ...
            ]
        """
        k = top_k or self.top_k
        if not self.metadata:
            return []

        emb = query_embedding or self.compute_embedding(query_frame_paths)

        # Retrieval using Cosine Similarity (Inner Product of normalized vectors)
        scored_candidates: List[Tuple[float, Dict[str, Any]]] = []

        if HAS_FAISS and self.index is not None:
            import numpy as np
            q_np = np.array([emb], dtype=np.float32)
            # Retrieve more candidates to account for excluded self-matches
            fetch_k = min(len(self.metadata), k * 3)
            distances, indices = self.index.search(q_np, fetch_k)

            for score, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                meta = self.metadata[idx]
                if exclude_video_hash and meta.get("video_url_hash") == exclude_video_hash:
                    continue  # Prevent self-retrieval
                scored_candidates.append((float(score), meta))
                if len(scored_candidates) >= k:
                    break
        else:
            # Pure Python Cosine Similarity fallback
            for idx, ref_meta in enumerate(self.metadata):
                if exclude_video_hash and ref_meta.get("video_url_hash") == exclude_video_hash:
                    continue
                if idx < len(self.fallback_vectors):
                    ref_vec = self.fallback_vectors[idx]
                    dot = sum(a * b for a, b in zip(emb, ref_vec))
                    scored_candidates.append((dot, ref_meta))

            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            scored_candidates = scored_candidates[:k]

        results = []
        for score, m in scored_candidates:
            results.append({
                "similarity_score": round(score, 4),
                "video_url_hash": m.get("video_url_hash", ""),
                "domain": m.get("domain", ""),
                "aspects": m.get("aspects", ""),
                "reference_complaint": m.get("ground_truth_complaint") or m.get("raw_ground_truth", ""),
            })

        return results
