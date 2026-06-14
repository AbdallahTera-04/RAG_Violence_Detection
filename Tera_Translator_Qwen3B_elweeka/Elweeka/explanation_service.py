"""Compatibility import for the Elweeka explanation service."""

from rag_service.explanation_service import NarratorClient, generate_clip_explanation


__all__ = ["NarratorClient", "generate_clip_explanation"]
