"""The audit log (§8 "Audit log", A-4/EU AI Act Art. 12 evidence): one SQLite
table, append-only, logging every inference and every clinician action. "One
database table. Costs an afternoon, buys a compliance section" is the spec's
own description of this component (§8) -- no new infrastructure beyond the
stdlib.

Append-only is enforced at the database level (triggers that abort any
UPDATE/DELETE), not just by convention -- a row written here cannot later be
edited or removed through this connection.
"""
from __future__ import annotations

import csv
import io
import json
import sqlite3
from datetime import datetime, timezone


def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            event_type TEXT NOT NULL,
            case_uid TEXT,
            input_hash TEXT,
            model_hash TEXT,
            payload_json TEXT NOT NULL
        )
    ''')
    conn.execute('''
        CREATE TRIGGER IF NOT EXISTS audit_log_no_update
        BEFORE UPDATE ON audit_log
        BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
    ''')
    conn.execute('''
        CREATE TRIGGER IF NOT EXISTS audit_log_no_delete
        BEFORE DELETE ON audit_log
        BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
    ''')
    conn.commit()
    return conn


def log_event(conn: sqlite3.Connection, event_type: str, case_uid: str | None = None,
              input_hash: str | None = None, model_hash: str | None = None, **payload) -> None:
    conn.execute(
        'INSERT INTO audit_log (ts, event_type, case_uid, input_hash, model_hash, payload_json) '
        'VALUES (?, ?, ?, ?, ?, ?)',
        (datetime.now(timezone.utc).isoformat(), event_type, case_uid, input_hash, model_hash,
         json.dumps(payload)),
    )
    conn.commit()


def fetch_recent(conn: sqlite3.Connection, limit: int = 200) -> list[dict]:
    cur = conn.execute(
        'SELECT id, ts, event_type, case_uid, input_hash, model_hash, payload_json '
        'FROM audit_log ORDER BY id DESC LIMIT ?', (limit,))
    rows = []
    for row in cur.fetchall():
        d = dict(zip(['id', 'ts', 'event_type', 'case_uid', 'input_hash', 'model_hash', 'payload_json'], row))
        d['payload'] = json.loads(d.pop('payload_json'))
        rows.append(d)
    return rows


def to_csv(conn: sqlite3.Connection) -> str:
    cur = conn.execute(
        'SELECT id, ts, event_type, case_uid, input_hash, model_hash, payload_json '
        'FROM audit_log ORDER BY id ASC')
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(['id', 'ts', 'event_type', 'case_uid', 'input_hash', 'model_hash', 'payload_json'])
    for row in cur.fetchall():
        writer.writerow(row)
    return buf.getvalue()
