# -*- coding: utf-8 -*-
"""backend/core/sqlite_queue.py

設計Aの Queue I/F（単機運用向け: SQLite）を BK33 に追加します。

目的:
- FastAPIプロセス（HTTP）から GPU 重い処理を切り離す（Worker側で実行）
- ジョブ状態の永続化（再起動耐性）
- シンプルで依存が少ない（Windows向け）

実装方針:
- 1ファイル SQLite で jobs テーブルを管理
- enqueue/dequeue/ack/heartbeat/get の最小 I/F を提供
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from lora_config import settings
from pathlib import Path
from backend.registry.index import register_job_output


def _now_ts() -> float:
    return time.time()


class SqliteQueue:
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = getattr(settings, "jobs_db_path", None)
        if db_path is None:
            db_path = settings.output_root / "jobs.db"
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.db_path), timeout=30, isolation_level=None)
        con.row_factory = sqlite3.Row
        return con

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    spec_hash TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    worker_id TEXT,
                    progress REAL,
                    result_json TEXT,
                    error TEXT
                );
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status_type ON jobs(status, job_type, created_at);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_jobs_spec_hash ON jobs(spec_hash);")

    # ---------------------------------------------------------------------
    # Queue I/F
    # ---------------------------------------------------------------------
    def enqueue(self, job_type: str, spec: Dict[str, Any], spec_hash: str) -> str:
        job_id = uuid.uuid4().hex
        now = _now_ts()
        with self._lock, self._conn() as con:
            con.execute(
                """
                INSERT INTO jobs(job_id, job_type, status, spec_json, spec_hash, created_at, updated_at, progress)
                VALUES(?,?,?,?,?,?,?,?)
                """,
                (job_id, job_type, "queued", json.dumps(spec, ensure_ascii=False), spec_hash, now, now, 0.0),
            )
        return job_id

    def dequeue(self, worker_id: str, job_type: str, timeout_sec: int = 0) -> Optional[Dict[str, Any]]:
        """最古の queued を running にして取得。

        timeout_sec は将来拡張用（ここではポーリングを前提として 0 を推奨）。
        """
        deadline = _now_ts() + max(0, int(timeout_sec))
        while True:
            with self._lock, self._conn() as con:
                con.execute("BEGIN IMMEDIATE;")
                row = con.execute(
                    """
                    SELECT job_id, job_type, spec_json, spec_hash, created_at
                      FROM jobs
                     WHERE status='queued' AND job_type=?
                     ORDER BY created_at ASC
                     LIMIT 1
                    """,
                    (job_type,),
                ).fetchone()
                if not row:
                    con.execute("COMMIT;")
                    if timeout_sec and _now_ts() < deadline:
                        time.sleep(0.3)
                        continue
                    return None

                now = _now_ts()
                con.execute(
                    """
                    UPDATE jobs
                       SET status='running', worker_id=?, started_at=?, updated_at=?
                     WHERE job_id=?
                    """,
                    (worker_id, now, now, row["job_id"]),
                )
                con.execute("COMMIT;")

                return {
                    "job_id": row["job_id"],
                    "job_type": row["job_type"],
                    "spec": json.loads(row["spec_json"]),
                    "spec_hash": row["spec_hash"],
                    "created_at": row["created_at"],
                }

    def heartbeat(self, job_id: str, progress: float) -> None:
        with self._lock, self._conn() as con:
            con.execute(
                "UPDATE jobs SET progress=?, updated_at=? WHERE job_id=?",
                (float(progress), _now_ts(), job_id),
            )

    def ack(self, job_id: str, status: str, result: Optional[Dict[str, Any]] = None, error: str = "") -> None:
        now = _now_ts()
        rj = json.dumps(result, ensure_ascii=False) if result is not None else None
        with self._lock, self._conn() as con:
            con.execute(
                """
                UPDATE jobs
                   SET status=?, result_json=?, error=?, finished_at=?, updated_at=?, progress=?
                 WHERE job_id=?
                """,
                (status, rj, error, now, now, 1.0 if status == "succeeded" else 0.0, job_id),
            )

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as con:
            row = con.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if not row:
            return None
        return {
            "job_id": row["job_id"],
            "job_type": row["job_type"],
            "status": row["status"],
            "spec_hash": row["spec_hash"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "worker_id": row["worker_id"],
            "progress": row["progress"],
            "spec": json.loads(row["spec_json"]),
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "error": row["error"] or "",
        }


sqlite_queue = SqliteQueue()