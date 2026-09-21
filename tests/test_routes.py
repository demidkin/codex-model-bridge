"""Regression checks for provider ownership, restore, and global config isolation."""

import asyncio
import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from bridge.claude import ClaudeEngine
from bridge.config import Settings
from bridge.registry import Registry
from bridge.router import Router


class FakeCore:
    """Small native-protocol fixture tracking all calls and persisted metadata."""

    def __init__(self):
        self.calls = []
        self.threads = {}
        self.config = {"model": "gpt-6-astra", "model_reasoning_effort": "xhigh"}

    async def call(self, method, params):
        """Emulate native metadata responses, never contacting a provider."""
        self.calls.append((method, copy.deepcopy(params)))
        await asyncio.sleep(0)
        if method in ("thread/start", "thread/resume"):
            tid = params.get("threadId") or f"task-{len(self.threads)}"
            thread = self.threads.get(tid, {"id": tid, "turns": []})
            thread.update({
                "model": params.get("model") or "gpt-6-astra",
                "modelProvider": params.get("modelProvider") or "openai",
                "cwd": params.get("cwd", "/tmp"),
            })
            self.threads[tid] = thread
            return {"thread": copy.deepcopy(thread), **{k: thread[k] for k in ("model", "modelProvider", "cwd")}}
        if method == "skills/list":
            return {"data":[{"cwd":cwd,"skills":[],"errors":[]} for cwd in params["cwds"]]}
        if method == "config/read":
            return {"config": copy.deepcopy(self.config), "origins": {}, "layers": [
                {"name": {"type": "user", "file": "/tmp/config.toml", "profile": None}, "version": "fixture"},
            ]}
        if method == "config/batchWrite":
            for edit in params["edits"]:
                self.config[edit["keyPath"]] = edit["value"]
            return {"status": "ok", "filePath": "/tmp/config.toml", "version": "fixture"}
        return {}


class RouteTests(unittest.IsolatedAsyncioTestCase):
    """Check cross-provider cases that could otherwise alter billing or lose context."""

    async def asyncSetUp(self):
        """Create isolated state and an explicitly fake private API-key file."""
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        key = root / "deepseek.key"
        key.write_text("fixture-key-never-used-over-network")
        key.chmod(0o600)
        self.settings = replace(Settings(), state=root, enable_deepseek=True, deepseek_key=key)
        self.registry = Registry(root / "routes.sqlite3")
        self.events = []

        async def emit(event):
            self.events.append(event)

        self.router = Router(self.settings, self.registry, emit)
        self.core = FakeCore()
        self.router.core = self.core

    async def asyncTearDown(self):
        """Remove only the fixture's own registry and fake key."""
        self.registry.close()
        self.temp.cleanup()

    async def new_task(self, **overrides):
        """Create a native task through the same routing entry point as the UI."""
        result = await self.router.request("thread/start", {"cwd": "/tmp", **overrides})
        return result["thread"]["id"]

    async def test_openai_is_unchanged(self):
        """Preserve native model/provider defaults and unknown request fields."""
        params = {"model": "gpt-6-astra", "unknownFutureOption": {"nested": True}}
        await self.router.request("thread/start", params)
        sent = self.core.calls[0][1]
        self.assertEqual({k: sent[k] for k in params}, params)
        self.assertNotIn("modelProvider", sent)
        self.assertEqual(sent["dynamicTools"][-1]["name"], "bridge_agents")

    async def test_first_turn_uses_collaboration_model_before_dispatch(self):
        """Move an empty task to DeepSeek before its first inference request."""
        tid = await self.new_task(approvalPolicy="never", sandbox="workspace-write")
        await self.router.request("turn/start", {
            "threadId": tid, "model": "gpt-6-astra", "input": [],
            "collaborationMode": {"mode": "default", "settings": {
                "model": "deepseek-flash", "reasoning_effort": "xhigh",
            }},
        })
        methods = [x[0] for x in self.core.calls]
        self.assertEqual(methods, ["thread/start", "skills/list", "thread/read", "thread/unsubscribe", "thread/resume", "turn/start"])
        resume = self.core.calls[-2][1]
        self.assertEqual(resume["modelProvider"], "bridge_deepseek")
        self.assertEqual(resume["approvalPolicy"], "never")
        self.assertEqual(resume["sandbox"], "workspace-write")
        self.assertNotIn("requires_openai_auth", resume["config"]["model_providers.bridge_deepseek"])
        self.assertEqual(self.core.calls[-1][1]["collaborationMode"]["settings"]["reasoning_effort"], "high")
        self.assertTrue(self.registry.get(tid)["locked"])

    async def test_locked_engine_cannot_switch(self):
        """Reject cross-engine continuation without sending an inference request."""
        tid = await self.new_task(model="gpt-6-astra")
        await self.router.request("turn/start", {"threadId": tid, "input": []})
        before = len(self.core.calls)
        with self.assertRaisesRegex(ValueError, "engine is fixed"):
            await self.router.request("turn/start", {"threadId": tid, "model": "deepseek-flash", "input": []})
        self.assertEqual(len(self.core.calls), before)

    async def test_restart_retains_provider_despite_global_model(self):
        """Restore from SQLite with the saved API provider, ignoring stale UI defaults."""
        tid = await self.new_task(model="deepseek-flash")
        await self.router.request("turn/start", {"threadId": tid, "input": []})
        self.registry.close()
        self.registry = Registry(self.settings.state / "routes.sqlite3")
        self.router.registry = self.registry
        await self.router.request("thread/resume", {"threadId": tid, "model": "gpt-6-astra"})
        params = self.core.calls[-1][1]
        self.assertEqual(params["modelProvider"], "bridge_deepseek")
        self.assertEqual(params["model"], "deepseek-flash")

    async def test_parallel_tasks_keep_separate_providers(self):
        """Concurrent tasks cannot overwrite each other's persisted route."""
        astra = await self.new_task(model="gpt-6-astra")
        deepseek = await self.new_task(model="deepseek-v4-pro")
        await asyncio.gather(*[
            self.router.request("turn/start", {"threadId": tid, "input": []})
            for tid in (astra, deepseek)
        ])
        self.assertEqual(self.registry.get(astra)["provider"], "openai")
        self.assertEqual(self.registry.get(deepseek)["provider"], "bridge_deepseek")

    async def test_private_key_required_before_network_dispatch(self):
        """Fail visibly when key permissions are wrong, without provider fallback."""
        tid = await self.new_task(model="deepseek-flash")
        self.settings.deepseek_key.chmod(0o644)
        before = len(self.core.calls)
        with self.assertRaisesRegex(ValueError, "0600"):
            await self.router.request("turn/start", {"threadId": tid, "input": []})
        self.assertEqual(len(self.core.calls), before)

    async def test_external_menu_preference_preserves_native_config(self):
        """Keep DeepSeek menu choice in the adapter, preserving OpenAI's global model."""
        await self.router.request("config/batchWrite", {"edits": [
            {"keyPath": "model", "value": "deepseek-flash", "mergeStrategy": "upsert"},
            {"keyPath": "model_reasoning_effort", "value": "high", "mergeStrategy": "upsert"},
        ]})
        self.assertEqual(self.core.config["model"], "gpt-6-astra")
        self.assertEqual(self.core.config["model_reasoning_effort"], "xhigh")
        self.assertFalse(any(method.startswith("config/") and "Write" in method for method, _ in self.core.calls))
        response = await self.router.request("config/read", {})
        self.assertEqual(response["config"]["model"], "deepseek-flash")

    async def test_external_menu_cannot_rewrite_project_configuration(self):
        """Reject a project-file target instead of copying effective defaults into it."""
        with self.assertRaisesRegex(ValueError, "default user configuration"):
            await self.router.request("config/value/write", {
                "keyPath": "model", "value": "deepseek-flash", "filePath": "/project/.codex/config.toml",
            })
        self.assertEqual(self.core.calls, [])

    async def test_external_menu_observes_native_config_version(self):
        """A stale UI cannot silently overwrite a concurrently changed preference."""
        with self.assertRaisesRegex(ValueError, "Configuration changed"):
            await self.router.request("config/value/write", {
                "keyPath": "model", "value": "deepseek-flash", "expectedVersion": "old-version",
            })
        self.assertIsNone(self.registry.preference("menu-model"))

    async def test_uncertain_dispatch_does_not_unlock_or_retry(self):
        """A transport failure after a turn was sent leaves its engine fixed."""
        tid = await self.new_task(model="deepseek-flash")
        original = self.core.call

        async def fail_turn(method, params):
            if method == "turn/start":
                raise ConnectionError("Disconnected after dispatch")
            return await original(method, params)

        self.core.call = fail_turn
        with self.assertRaises(ConnectionError):
            await self.router.request("turn/start", {"threadId": tid, "input": []})
        self.assertTrue(self.registry.get(tid)["locked"])

    async def test_ephemeral_switch_is_rejected_without_unsubscribe(self):
        """Do not destroy an ephemeral task that cannot be resumed from disk."""
        tid = await self.new_task(ephemeral=True)
        before = len(self.core.calls)
        with self.assertRaisesRegex(ValueError, "ephemeral"):
            await self.router.request("turn/start", {"threadId": tid, "model": "deepseek-flash", "input": []})
        self.assertEqual(len(self.core.calls), before)

    async def test_native_empty_thread_materialization_error_is_compatible(self):
        """Tolerate only the exact legacy-history error from the tested core build."""
        from bridge.rpc import RpcError
        tid = await self.new_task()
        original = self.core.call

        async def legacy_history(method, params):
            if method == "thread/read":
                raise RpcError({"code": -32601, "message": "list_turns is not supported yet"})
            return await original(method, params)

        self.core.call = legacy_history
        await self.router.request("turn/start", {"threadId": tid, "model": "deepseek-flash", "input": []})
        self.assertEqual(self.registry.get(tid)["provider"], "bridge_deepseek")

    async def test_initial_deepseek_selection_can_return_to_openai_before_turn(self):
        """Never leak the DeepSeek-only catalogue into an empty task switched to Astra."""
        tid = await self.new_task(model="deepseek-flash")
        await self.router.request("turn/start", {"threadId": tid, "model": "gpt-6-astra", "input": []})
        resume = next(params for method, params in self.core.calls if method == "thread/resume")
        self.assertEqual(resume["modelProvider"], "openai")
        self.assertNotIn("model_catalog_json", resume["config"])

    async def test_unfinished_history_is_not_replayed(self):
        """Crash recovery changes status only and performs no engine calls."""
        self.registry.save_turn("fixture", {"id": "turn", "items": [{"type":"commandExecution", "status":"inProgress"}], "status": "inProgress"})
        self.registry.recover()
        self.assertEqual(self.registry.turns("fixture")[0]["status"], "interrupted")
        self.assertEqual(self.registry.turns("fixture")[0]["items"][0]["status"], "failed")
        self.assertEqual(self.core.calls, [])

    async def test_ui_catalog_bypasses_filter_without_native_override(self):
        response = await self.router.request("config/read", {})
        hint = response["config"]["model_catalog_json"]
        self.assertTrue(Path(hint).exists())
        self.assertNotIn("model_catalog_json", self.core.config)
        await self.router.request("thread/start", {"model":"gpt-6-astra", "config":{"model_catalog_json":hint}})
        self.assertNotIn("model_catalog_json", self.core.calls[-1][1]["config"])

    async def test_resume_preserves_workspace_and_permissions(self):
        tid = await self.new_task(model="deepseek-flash", cwd="/fixture/workspace", approvalPolicy="untrusted", sandbox="workspace-write")
        await self.router.request("thread/resume", {"threadId":tid, "cwd":None})
        params = self.core.calls[-1][1]
        self.assertEqual(params["cwd"], "/fixture/workspace")
        self.assertEqual(params["sandbox"], "workspace-write")
        self.assertEqual(params["approvalPolicy"], "untrusted")

    async def test_external_effort_only_edit_preserves_openai_defaults(self):
        self.registry.preference("menu-model", {"model":"deepseek-flash"})
        await self.router.request("config/value/write", {"keyPath":"model_reasoning_effort", "value":"low", "mergeStrategy":"upsert"})
        self.assertEqual(self.core.config["model_reasoning_effort"], "xhigh")
        self.assertEqual(self.registry.preference("menu-model")["model_reasoning_effort"], "low")

    async def test_resumed_claude_thread_gets_a_cursor_only_once_it_has_turns(self):
        """Native core never saw a Claude turn, so it always answers thread/resume
        with null pagination cursors; the desktop reads that as "no history" and
        never follows up with thread/turns/list or thread/items/list, which is
        what left every restored Claude task looking empty after a restart."""
        self.router.claude = ClaudeEngine(self.settings, self.registry, self.router.emit, self.core)
        self.registry.save({"thread_id": "fixture", "engine": "claude", "provider": "bridge_claude",
                             "model": "claude-code/sonnet", "cwd": "/tmp", "options": {}, "locked": True})
        empty = await self.router.request("thread/resume", {"threadId": "fixture", "excludeTurns": True})
        self.assertIsNone(empty.get("turnsBackwardsCursor"))
        self.assertIsNone(empty.get("itemsBackwardsCursor"))
        self.registry.save_turn("fixture", {"id": "t1", "status": "completed", "startedAt": 1,
                                             "items": [{"type": "userMessage", "id": "u1",
                                                        "content": [{"type": "text", "text": "hi"}]}]})
        nonempty = await self.router.request("thread/resume", {"threadId": "fixture", "excludeTurns": True})
        self.assertEqual(nonempty["turnsBackwardsCursor"], "0")
        self.assertEqual(nonempty["itemsBackwardsCursor"], "0")
