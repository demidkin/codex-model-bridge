"""Private durable route ownership; no keys and no OpenAI conversation copies."""

import json
import os
import sqlite3
import time
from pathlib import Path
from .leases import Lease


class Registry:
    """Own SQLite registry, independent from Codex's internal databases."""

    def __init__(self, path):
        # umask is set at process entry, before creating SQLite and its journals.
        self.state = Path(path).parent
        self.db = sqlite3.connect(path, timeout=10)
        os.chmod(path, 0o600)
        self.db.row_factory = sqlite3.Row
        # WAL lets concurrent bridge processes (one per open task/host connection)
        # read this shared registry while another one is mid-write, instead of
        # blocking behind the default rollback-journal writer lock.
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=10000")
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1, 2, 3, 4):
            raise ValueError("Unsupported bridge registry version")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS routes (
                thread_id TEXT PRIMARY KEY,
                engine TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                cwd TEXT NOT NULL,
                locked INTEGER NOT NULL DEFAULT 0,
                options TEXT NOT NULL DEFAULT '{}',
                session_id TEXT,
                updated_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS claude_turns (
                thread_id TEXT NOT NULL,
                turn_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS preferences (
                name TEXT PRIMARY KEY, value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS claude_threads (
                thread_id TEXT PRIMARY KEY, payload TEXT NOT NULL,
                archived INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY, parent_id TEXT NOT NULL, payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS agent_calls (
                call_key TEXT PRIMARY KEY, payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS task_capabilities (
                thread_id TEXT PRIMARY KEY, payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS task_instructions (
                thread_id TEXT PRIMARY KEY, instructions TEXT NOT NULL
            );
            PRAGMA user_version=4;
        """)

    def get(self, thread_id):
        """Return a persisted route, or None for a task the bridge has not seen."""
        row = self.db.execute(
            "SELECT * FROM routes WHERE thread_id=?", (thread_id,),
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["options"] = json.loads(result["options"])
        return result

    def save(self, route):
        """Atomically persist route selection before dispatching a turn."""
        with self.db:
            self.db.execute(
                """INSERT INTO routes VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(thread_id) DO UPDATE SET
                engine=excluded.engine, provider=excluded.provider,
                model=excluded.model, cwd=excluded.cwd, locked=excluded.locked,
                options=excluded.options, session_id=excluded.session_id,
                updated_at=excluded.updated_at""",
                (
                    route["thread_id"], route["engine"], route["provider"],
                    route["model"], route["cwd"], int(route.get("locked", False)),
                    json.dumps(route.get("options", {})), route.get("session_id"),
                    int(time.time()),
                ),
            )

    def save_turn(self, thread_id, turn):
        """Store Claude UI history, separate from protocol diagnostic logs."""
        with self.db:
            self.db.execute(
                """INSERT INTO claude_turns VALUES (?,?,?,?)
                ON CONFLICT(turn_id) DO UPDATE SET payload=excluded.payload""",
                (thread_id, turn["id"], json.dumps(turn), int(time.time())),
            )

    def turns(self, thread_id):
        """Load this task's Claude history in creation order."""
        rows = self.db.execute(
            "SELECT payload FROM claude_turns WHERE thread_id=? ORDER BY created_at,rowid",
            (thread_id,),
        )
        return [json.loads(row[0]) for row in rows]

    def save_thread(self, thread, archived=None):
        """Persist Claude sidebar metadata without touching native Codex databases."""
        with self.db:
            self.db.execute(
                "INSERT INTO claude_threads(thread_id,payload) VALUES (?,?) "
                "ON CONFLICT(thread_id) DO UPDATE SET payload=excluded.payload",
                (thread['id'], json.dumps(thread)),
            )
            if archived is not None:
                self.db.execute('UPDATE claude_threads SET archived=? WHERE thread_id=?',
                                (int(archived), thread['id']))

    def set_archived(self, thread_id, archived):
        with self.db:
            self.db.execute('UPDATE claude_threads SET archived=? WHERE thread_id=?', (int(archived), thread_id))

    def claude_threads(self, archived=False):
        rows = self.db.execute('SELECT payload FROM claude_threads WHERE archived=?', (int(archived),))
        return [json.loads(row[0]) for row in rows]

    def recover(self):
        """Mark unfinished Claude turns interrupted; do not replay their actions."""
        rows = self.db.execute("SELECT thread_id,payload FROM claude_turns").fetchall()
        for row in rows:
            turn = json.loads(row["payload"])
            if turn["status"] == "inProgress":
                lease = Lease(self.state, "claude:" + row["thread_id"])
                if not lease.acquire():
                    continue
                try:
                    # Re-read under the lease: another owner may have just completed.
                    fresh = self.db.execute("SELECT payload FROM claude_turns WHERE turn_id=?", (turn["id"],)).fetchone()
                    turn = json.loads(fresh[0])
                    if turn["status"] != "inProgress":
                        continue
                    turn["status"] = "interrupted"
                    turn["completedAt"] = int(time.time())
                    for item in turn.get("items", []):
                        if item.get("status") == "inProgress":
                            item["status"] = "failed"
                    self.save_turn(row["thread_id"], turn)
                finally:
                    lease.close()

    def agent(self, ident):
        row = self.db.execute('SELECT payload FROM agents WHERE id=?', (ident,)).fetchone()
        return json.loads(row[0]) if row else None

    def capabilities(self, thread_id, value=None):
        """Only hosting tool schemas and selected roots, never auth configuration."""
        if value is not None:
            safe = {k: value[k] for k in ('dynamicTools', 'selectedCapabilityRoots') if k in value}
            with self.db:
                self.db.execute('INSERT INTO task_capabilities VALUES (?,?) ON CONFLICT(thread_id) DO UPDATE SET payload=excluded.payload',
                                (thread_id, json.dumps(safe)))
        row = self.db.execute('SELECT payload FROM task_capabilities WHERE thread_id=?', (thread_id,)).fetchone()
        return json.loads(row[0]) if row else {}

    def task_instructions(self, thread_id, value=None):
        """Keep a task's caller instructions when refreshing its skill catalogue.

        None means an older task whose instructions are owned by native Codex;
        it must not be overwritten with a catalogue alone. Never inherit this
        text into children implicitly.
        """
        if value is not None:
            with self.db:
                self.db.execute('INSERT INTO task_instructions VALUES (?,?) ON CONFLICT(thread_id) DO UPDATE SET instructions=excluded.instructions',
                                (thread_id, value))
        row = self.db.execute('SELECT instructions FROM task_instructions WHERE thread_id=?', (thread_id,)).fetchone()
        return row[0] if row else None

    def agents(self, parent_id=None):
        rows = self.db.execute('SELECT payload FROM agents' + (' WHERE parent_id=?' if parent_id else ''),
                               (parent_id,) if parent_id else ())
        return [json.loads(row[0]) for row in rows]

    def save_agent(self, agent):
        with self.db:
            self.db.execute('INSERT INTO agents VALUES (?,?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload',
                            (agent['id'], agent['parent_id'], json.dumps(agent)))

    def agent_call(self, key, result=None):
        if result is not None:
            with self.db:
                self.db.execute('INSERT INTO agent_calls VALUES (?,?) ON CONFLICT(call_key) DO UPDATE SET payload=excluded.payload',
                                (key,json.dumps(result)))
        row = self.db.execute('SELECT payload FROM agent_calls WHERE call_key=?',(key,)).fetchone()
        return json.loads(row[0]) if row else None

    def preference(self, name, value=None):
        """Read or atomically write a nonsecret adapter preference."""
        if value is not None:
            with self.db:
                self.db.execute(
                    "INSERT INTO preferences VALUES (?,?) ON CONFLICT(name) DO UPDATE SET value=excluded.value",
                    (name, json.dumps(value)),
                )
        row = self.db.execute("SELECT value FROM preferences WHERE name=?", (name,)).fetchone()
        return json.loads(row[0]) if row else None

    def close(self):
        """Flush and close the adapter's registry."""
        self.db.close()
