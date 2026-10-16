from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from database.writer import DatabaseWriter
from utils.config import ensure_project_dirs


def main() -> None:
    ensure_project_dirs()
    writer = DatabaseWriter()
    writer.initialize()
    print(f"Database initialized: {writer.database_path}")


if __name__ == "__main__":
    main()

