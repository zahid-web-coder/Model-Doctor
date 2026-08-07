"""Reusable helpers shared across Model Doctor.

Anything used by more than one module belongs here rather than being copied.
Kept deliberately free of re-exports so that imports name their source module
explicitly (``from utils.resources import find_images``), which keeps the
dependency graph readable as the project grows.
"""
