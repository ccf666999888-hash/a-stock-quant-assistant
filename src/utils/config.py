from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AppPaths(BaseModel):
    root: Path = PROJECT_ROOT

    @property
    def database(self) -> Path:
        return self.root / "data" / "db" / "a_stock_quant.duckdb"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    if not candidate.exists():
        return {}
    with candidate.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_env_file(path: str | Path = ".env") -> None:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    if not candidate.exists():
        return
    for raw_line in candidate.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def ensure_project_dirs() -> None:
    for relative in (
        "data/raw",
        "data/processed",
        "data/cache",
        "data/db",
        "outputs/reports",
        "outputs/stock_pools",
        "outputs/backtests",
        "outputs/charts",
    ):
        (PROJECT_ROOT / relative).mkdir(parents=True, exist_ok=True)

