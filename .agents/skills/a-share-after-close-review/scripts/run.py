from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]
ENGINE = PROJECT / "scripts" / "generate_after_close_html.py"
VENV_PYTHON = PROJECT / ".venv" / "Scripts" / "python.exe"


def main() -> None:
    if not ENGINE.exists():
        raise FileNotFoundError(f"After-close engine not found: {ENGINE}")
    python = VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)
    subprocess.run([str(python), str(ENGINE), *sys.argv[1:]], cwd=PROJECT, check=True)


if __name__ == "__main__":
    main()

