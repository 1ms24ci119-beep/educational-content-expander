"""Compatibility facade for the local n-gram generator.

The project intentionally uses a transparent statistical language model rather
than a remote LLM, so it can run offline and be used in introductory lessons.
"""
from .core import NGramModel

def build_generator(texts, n: int = 2, k: float = 0.5, seed: int | None = None) -> NGramModel:
    """Train and return a ready-to-use local generator."""
    return NGramModel(n=n, k=k, seed=seed).fit(texts)
