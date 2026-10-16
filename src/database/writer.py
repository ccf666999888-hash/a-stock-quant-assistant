from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
from loguru import logger

from database.schema import ALL_TABLE_SQL, MIGRATION_SQL
from utils.config import AppPaths, ensure_project_dirs


class DatabaseWriter:
    def __init__(self, database_path: str | Path | None = None):
        ensure_project_dirs()
        self.database_path = Path(database_path or AppPaths().database)

    def connect(self):
        return duckdb.connect(str(self.database_path))

    def initialize(self) -> None:
        with self.connect() as connection:
            for sql in ALL_TABLE_SQL.values():
                connection.execute(sql)
            for sql in MIGRATION_SQL:
                connection.execute(sql)
        logger.info("Initialized DuckDB at {}", self.database_path)

    def append_dataframe(self, table: str, frame: pd.DataFrame) -> int:
        if frame.empty:
            logger.warning("Skipping empty write to {}", table)
            return 0
        if table not in ALL_TABLE_SQL:
            raise ValueError(f"Unknown table: {table}")
        self.initialize()
        with self.connect() as connection:
            target_columns = {
                row[1] for row in connection.execute(f"PRAGMA table_info('{table}')").fetchall()
            }
            usable = [column for column in frame.columns if column in target_columns]
            if not usable:
                logger.warning("No compatible columns for {}; write skipped", table)
                return 0
            payload = frame[usable].copy()
            connection.register("incoming_frame", payload)
            connection.execute(f"INSERT INTO {table} BY NAME SELECT * FROM incoming_frame")
        return len(frame)

    def upsert_dataframe(
        self,
        table: str,
        frame: pd.DataFrame,
        keys: tuple[str, ...] = ("ts_code", "trade_date"),
    ) -> int:
        if frame.empty:
            return 0
        self.initialize()
        available_keys = [key for key in keys if key in frame.columns]
        with self.connect() as connection:
            if available_keys:
                key_frame = frame[available_keys].drop_duplicates()
                connection.register("incoming_keys", key_frame)
                conditions = " AND ".join(
                    f"target.{key} = incoming_keys.{key}" for key in available_keys
                )
                connection.execute(
                    f"DELETE FROM {table} AS target USING incoming_keys WHERE {conditions}"
                )
        return self.append_dataframe(table, frame)
