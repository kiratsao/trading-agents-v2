"""Backward-compat shim — Settings now live in core.config.settings."""

from core.config.settings import Settings, settings  # noqa: F401

__all__ = ["Settings", "settings"]
