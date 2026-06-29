"""Image embedder interface (Phase 2 visual search).

Default is a dependency-light ``ColorGridEmbedder`` (downsampled colour grid — no torch),
so the visual endpoint and the sanity eval run everywhere, including CI and the free-tier
deploy. ``OpenClipEmbedder`` is the genuine multimodal upgrade (real CLIP) behind the opt-in
``[clip]`` extra. Both embed into an L2-normalized space so cosine = dot product.

Grounding does not depend on the image embedder: it is a *retrieval* signal only; every
factual claim in a visual recommendation's rationale is still templated from the record and
audited by the GroundingVerifier.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from PIL import Image

from core.config import Settings, get_settings


class ImageEmbedder(Protocol):
    def embed_images(self, paths: list[str]) -> np.ndarray: ...  # (n, dim) float32, L2-norm

    @property
    def dim(self) -> int: ...

    @property
    def name(self) -> str: ...


def _l2(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, 1e-9, None)


class ColorGridEmbedder:
    """Downsampled colour-grid features: resize to GxG, flatten RGB, L2-normalize.

    Captures the renders' dominant colours (metal/stone) and coarse layout (category
    silhouette), so visually-similar synthetic renders land near each other. Deterministic,
    no torch — runs in CI and on the free-tier deploy.
    """

    def __init__(self, grid: int = 8):
        self._grid = grid

    @property
    def dim(self) -> int:
        return self._grid * self._grid * 3

    @property
    def name(self) -> str:
        return f"color-grid-{self._grid}"

    def embed_images(self, paths: list[str]) -> np.ndarray:
        out = np.zeros((len(paths), self.dim), dtype=np.float32)
        for i, path in enumerate(paths):
            with Image.open(path) as img:
                small = img.convert("RGB").resize((self._grid, self._grid), Image.BILINEAR)
            out[i] = (np.asarray(small, dtype=np.float32) / 255.0).reshape(-1)
        return _l2(out)


class _OpenClipEmbedder:  # pragma: no cover - optional heavy extra (torch)
    """Real CLIP via open_clip (multimodal: shared image+text space)."""

    # ViT-B-32-quickgelu matches the openai checkpoint's activation (plain ViT-B-32 warns and
    # slightly degrades it). Same cached weights — no extra download.
    def __init__(self, model_name: str = "ViT-B-32-quickgelu", pretrained: str = "openai"):
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise ImportError("Install the 'clip' extra: pip install '.[clip]'") from exc
        self._torch = torch
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self._name = f"clip-{model_name}-{pretrained}"

    @property
    def dim(self) -> int:
        return int(self._model.visual.output_dim)

    @property
    def name(self) -> str:
        return self._name

    def embed_images(self, paths: list[str]) -> np.ndarray:
        torch = self._torch
        tensors = []
        for path in paths:
            with Image.open(path) as img:
                tensors.append(self._preprocess(img.convert("RGB")))
        batch = torch.stack(tensors)
        with torch.no_grad():
            feats = self._model.encode_image(batch)
        return _l2(feats.cpu().numpy().astype(np.float32))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        tokens = self._tokenizer(texts)
        with torch.no_grad():
            feats = self._model.encode_text(tokens)
        return _l2(feats.cpu().numpy().astype(np.float32))


def get_image_embedder(settings: Settings | None = None) -> ImageEmbedder:
    settings = settings or get_settings()
    if settings.image_embedder == "clip":
        return _OpenClipEmbedder()
    return ColorGridEmbedder()
