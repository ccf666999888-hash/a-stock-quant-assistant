from __future__ import annotations

from pathlib import Path


def write_markdown(path: str | Path, content: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content.rstrip() + "\n", encoding="utf-8")
    return destination

