"""Live dashboard — Rich terminal or Streamlit web UI."""

from __future__ import annotations


class Dashboard:
    """Renders live portfolio state. Backend is selected by config (rich | streamlit)."""

    def __init__(self, db_path: str, backend: str = "rich") -> None:
        self.db_path = db_path
        self.backend = backend

    def start(self) -> None:
        """Launch the dashboard loop."""
        raise NotImplementedError

    def _render_rich(self) -> None:
        """Terminal dashboard using Rich Live."""
        raise NotImplementedError

    def _render_streamlit(self) -> None:
        """Web dashboard using Streamlit."""
        raise NotImplementedError
