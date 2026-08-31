from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class ResponseStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS scans (
                    scan_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS actions (
                    action_id TEXT PRIMARY KEY, plan_id TEXT UNIQUE NOT NULL, state TEXT NOT NULL,
                    actor TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, expires_at TEXT,
                    native_identifiers_json TEXT NOT NULL DEFAULT '[]', verification_json TEXT,
                    failure_json TEXT
                );
                CREATE TABLE IF NOT EXISTS plans (
                    plan_id TEXT PRIMARY KEY, action_id TEXT UNIQUE NOT NULL, plan_hash TEXT NOT NULL,
                    scan_fingerprint TEXT NOT NULL, created_at TEXT NOT NULL, payload_json TEXT NOT NULL,
                    FOREIGN KEY(action_id) REFERENCES actions(action_id)
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT, action_id TEXT NOT NULL,
                    state TEXT NOT NULL, actor TEXT, created_at TEXT NOT NULL, payload_json TEXT NOT NULL,
                    FOREIGN KEY(action_id) REFERENCES actions(action_id)
                );
                CREATE INDEX IF NOT EXISTS idx_events_action ON events(action_id, event_id);
                CREATE INDEX IF NOT EXISTS idx_actions_state ON actions(state, updated_at);
                """
            )
        os.chmod(self.path, 0o600)
        for suffix in ("-wal", "-shm"):
            companion = Path(str(self.path) + suffix)
            if companion.exists():
                os.chmod(companion, 0o600)

    def save_scan(self, scan_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        created = utcnow()
        with self.connect() as db:
            db.execute("INSERT INTO scans VALUES (?, ?, ?)", (scan_id, created, _dump(payload)))
        return {"scan_id": scan_id, "created_at": created, **payload}

    def get_scan(self, scan_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM scans WHERE scan_id=?", (scan_id,)).fetchone()
        return None if row is None else {"scan_id": row["scan_id"], "created_at": row["created_at"], **json.loads(row["payload_json"])}

    def create_plan_action(
        self, *, plan_id: str, action_id: str, plan_hash: str, scan_fingerprint: str,
        payload: dict[str, Any], actor: str | None,
    ) -> None:
        now = utcnow()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                db.execute(
                    "INSERT INTO actions(action_id,plan_id,state,actor,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                    (action_id, plan_id, "PROPOSED", actor, now, now),
                )
                db.execute("INSERT INTO plans VALUES(?,?,?,?,?,?)",
                           (plan_id, action_id, plan_hash, scan_fingerprint, now, _dump(payload)))
                db.execute("INSERT INTO events(action_id,state,actor,created_at,payload_json) VALUES(?,?,?,?,?)",
                           (action_id, "DETECTED", actor, now, _dump({"evidence": payload.get("evidence", {})})))
                db.execute("INSERT INTO events(action_id,state,actor,created_at,payload_json) VALUES(?,?,?,?,?)",
                           (action_id, "PROPOSED", actor, now, _dump({"plan_hash": plan_hash})))
                db.commit()
            except Exception:
                db.rollback()
                raise

    def get_plan(self, plan_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM plans WHERE plan_id=?", (plan_id,)).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload_json"])
        return {**payload, "plan_id": row["plan_id"], "action_id": row["action_id"],
                "plan_hash": row["plan_hash"], "scan_fingerprint": row["scan_fingerprint"],
                "created_at": row["created_at"]}

    def transition(
        self, action_id: str, *, expected: Iterable[str], state: str, actor: str | None = None,
        event_payload: dict[str, Any] | None = None, native_identifiers: list[str] | None = None,
        verification: dict[str, Any] | None = None, failure: dict[str, Any] | None = None,
        expires_at: str | None = None,
    ) -> bool:
        expected_values = tuple(expected)
        if not expected_values:
            return False
        now = utcnow()
        assignments = ["state=?", "updated_at=?"]
        values: list[Any] = [state, now]
        if actor is not None:
            assignments.append("actor=?")
            values.append(actor)
        if native_identifiers is not None:
            assignments.append("native_identifiers_json=?")
            values.append(_dump(native_identifiers))
        if verification is not None:
            assignments.append("verification_json=?")
            values.append(_dump(verification))
        if failure is not None:
            assignments.append("failure_json=?")
            values.append(_dump(failure))
        if expires_at is not None:
            assignments.append("expires_at=?")
            values.append(expires_at)
        placeholders = ",".join("?" for _ in expected_values)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            cursor = db.execute(
                f"UPDATE actions SET {','.join(assignments)} WHERE action_id=? AND state IN ({placeholders})",
                (*values, action_id, *expected_values),
            )
            if cursor.rowcount != 1:
                db.rollback()
                return False
            db.execute("INSERT INTO events(action_id,state,actor,created_at,payload_json) VALUES(?,?,?,?,?)",
                       (action_id, state, actor, now, _dump(event_payload or {})))
            db.commit()
        return True

    def set_expires_at(self, action_id: str, expires_at: str) -> None:
        with self.connect() as db:
            db.execute("UPDATE actions SET expires_at=?, updated_at=? WHERE action_id=?", (expires_at, utcnow(), action_id))

    def get_action(self, action_id: str, *, include_events: bool = True) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM actions WHERE action_id=?", (action_id,)).fetchone()
            events = db.execute("SELECT state,actor,created_at,payload_json FROM events WHERE action_id=? ORDER BY event_id",
                                (action_id,)).fetchall() if row and include_events else []
        if row is None:
            return None
        value = dict(row)
        value["native_identifiers"] = json.loads(value.pop("native_identifiers_json") or "[]")
        value["verification"] = json.loads(value.pop("verification_json")) if value["verification_json"] else None
        value["failure"] = json.loads(value.pop("failure_json")) if value["failure_json"] else None
        value["events"] = [{"state": item["state"], "actor": item["actor"], "created_at": item["created_at"],
                            "payload": json.loads(item["payload_json"])} for item in events]
        return value

    def list_actions(self, states: Iterable[str] | None = None) -> list[dict[str, Any]]:
        with self.connect() as db:
            if states:
                values = tuple(states)
                placeholders = ",".join("?" for _ in values)
                rows = db.execute(f"SELECT action_id FROM actions WHERE state IN ({placeholders}) ORDER BY created_at DESC", values).fetchall()
            else:
                rows = db.execute("SELECT action_id FROM actions ORDER BY created_at DESC").fetchall()
        return [self.get_action(row["action_id"], include_events=False) for row in rows]
