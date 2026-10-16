from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from utils.config import AppPaths


class DatabaseReader:
    def __init__(self, database_path: str | Path | None = None):
        self.database_path = Path(database_path or AppPaths().database)

    def query(self, sql: str, params: list | tuple | None = None) -> pd.DataFrame:
        if not self.database_path.exists():
            return pd.DataFrame()
        with duckdb.connect(str(self.database_path), read_only=True) as connection:
            return connection.execute(sql, params or []).fetchdf()

    def table(
        self,
        table: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        ts_codes: list[str] | None = None,
    ) -> pd.DataFrame:
        clauses: list[str] = []
        params: list[str] = []
        if start_date:
            clauses.append("trade_date >= ?")
            params.append(start_date)
        if end_date:
            clauses.append("trade_date <= ?")
            params.append(end_date)
        if ts_codes:
            placeholders = ",".join("?" for _ in ts_codes)
            clauses.append(f"ts_code IN ({placeholders})")
            params.extend(ts_codes)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        try:
            return self.query(f"SELECT * FROM {table}{where}", params)
        except duckdb.Error:
            return pd.DataFrame()

