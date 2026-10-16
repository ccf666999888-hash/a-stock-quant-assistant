from __future__ import annotations

import importlib.util
from pathlib import Path


def load_sync_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "sync_tushare_data.py"
    spec = importlib.util.spec_from_file_location("sync_tushare_data", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_date_chunks_cover_range_without_overlap():
    module = load_sync_module()
    chunks = module.date_chunks("20260101", "20260131", 14)
    assert chunks == [
        ("20260101", "20260114"),
        ("20260115", "20260128"),
        ("20260129", "20260131"),
    ]

