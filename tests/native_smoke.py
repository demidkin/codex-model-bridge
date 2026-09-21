"""Opt-in native-engine checks: simulated DeepSeek, optionally real subscription Astra."""

import argparse
import asyncio
import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from bridge.config import Settings
from bridge.registry import Registry
from bridge.router import Router
from bridge.rpc import CoreClient


class ResponsesFixture(BaseHTTPRequestHandler):
    """Serve a deterministic Responses stream and record only routing metadata."""

    requests = []

    def log_message(self, *args):
        """Disable default access logging for the local API fixture."""

    def do_POST(self):
        """Return one completed assistant message in OpenAI Responses SSE format."""
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.requests.append({
            "path": self.path, "model": body.get("model"),
            "fixture_auth": self.headers.get("Authorization") == "Bearer bridge-fixture-key",
            "tool_count": len(body.get("tools", [])),
        })
        item = {"id": "msg_fixture", "type": "message", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": "DEEPSEEK_FIXTURE_OK", "annotations": []}]}
        response = {"id": "resp_fixture", "object": "response", "status": "completed",
                    "model": body["model"], "output": [item],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                              "input_tokens_details": {"cached_tokens": 0},
                              "output_tokens_details": {"reasoning_tokens": 0}}}
        events = [
            {"type": "response.created", "response": {**response, "status": "in_progress", "output": []}},
            {"type": "response.output_item.added", "output_index": 0,
             "item": {**item, "status": "in_progress", "content": []}},
            {"type": "response.output_text.delta", "item_id": item["id"], "output_index": 0,
             "content_index": 0, "delta": "DEEPSEEK_FIXTURE_OK"},
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {"type": "response.completed", "response": response},
        ]
        payload = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


async def run(live_astra):
    """Exercise first-turn routing and restart against the real bundled app-server."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), ResponsesFixture)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    class FixtureSettings(Settings):
        """Override only the test provider URL; no production proxy is installed."""

        def provider_config(self, engine):
            """Keep real provider configuration, sending test inference to loopback."""
            config = super().provider_config(engine)
            if engine == "deepseek":
                config["model_providers.bridge_deepseek"]["base_url"] = f"http://127.0.0.1:{server.server_port}"
            return config

    report = {"live_astra_requested": live_astra, "deepseek_api": "local fixture only"}
    with tempfile.TemporaryDirectory(prefix="codex-bridge-native-") as temporary:
        root = Path(temporary)
        key = root / "fixture.key"
        key.write_text("bridge-fixture-key")
        key.chmod(0o600)
        settings = FixtureSettings(state=root, deepseek_key=key, enable_deepseek=True)
        registry = Registry(root / "routes.sqlite3")
        events = []
        completed = {}
        turn_errors = []

        async def emit(event):
            events.append(event)
            if event.get("method") == "turn/completed":
                params = event["params"]
                completed[params["threadId"]] = params["turn"]
            if event.get("method") == "error":
                turn_errors.append(event["params"]["error"].get("message", "error"))
            if "id" in event and "method" in event:
                await router.core.send({"id": event["id"], "result": {"decision": "decline"}})

        async def start_core():
            router = Router(settings, registry, emit)
            core = CoreClient([
                settings.core, "app-server", "-c", "mcp_servers.unityMCP.enabled=false",
                "-c", "mcp_servers.node_repl.enabled=false", "-c", "mcp_servers.computer-use.enabled=false",
            ], router.event)
            router.core = core
            await core.start()
            await router.request("initialize", {"clientInfo": {"name": "bridge_native_smoke", "version": "0.1"},
                                                 "capabilities": {"experimentalApi": True}})
            await core.send({"method": "initialized"})
            return router, core

        async def turn(tid, model, prompt):
            completed.pop(tid, None)
            response = await router.request("turn/start", {"threadId": tid, "model": model, "effort": "low",
                "input": [{"type": "text", "text": prompt, "text_elements": []}]})
            async with asyncio.timeout(90):
                while tid not in completed:
                    await asyncio.sleep(0.05)
            result = completed[tid]
            if result["status"] != "completed":
                raise RuntimeError(f"Native turn failed: {result.get('error')} {turn_errors}")
            return response

        router, core = await start_core()
        tid = None
        try:
            models = await router.request("model/list", {"includeHidden": True, "limit": 100})
            report["catalogue"] = [model["model"] for model in models["data"]]
            initial = await router.request("thread/start", {
                "model": "gpt-6-astra", "cwd": str(root), "ephemeral": False,
                "approvalPolicy": "never", "sandbox": "read-only", "experimentalRawEvents": False,
            })
            tid = initial["thread"]["id"]
            report["fixture_thread_id"] = tid
            await turn(tid, "deepseek-flash", "Reply with the fixture acknowledgement. Do not use tools.")
            report["first_turn_provider"] = registry.get(tid)["provider"]
            await core.close()
            router, core = await start_core()
            resumed = await router.request("thread/resume", {"threadId": tid, "model": "gpt-6-astra", "excludeTurns": True})
            report["resumed_provider"] = resumed["modelProvider"]
            await turn(tid, "deepseek-v4-pro", "Reply with the fixture acknowledgement. Do not use tools.")
            report["fixture_requests"] = ResponsesFixture.requests
            assert [r["model"] for r in ResponsesFixture.requests] == ["deepseek-flash", "deepseek-v4-pro"]
            assert all(r["fixture_auth"] for r in ResponsesFixture.requests)
            if live_astra:
                astra = await router.request("thread/start", {
                    "model": "gpt-6-astra", "cwd": str(root), "ephemeral": True,
                    "approvalPolicy": "never", "sandbox": "read-only", "experimentalRawEvents": False,
                })
                aid = astra["thread"]["id"]
                await turn(aid, "gpt-6-astra", "Run exactly one shell command: /usr/bin/printf 'BRIDGE_TOOL_OK\\n'. Then respond exactly BRIDGE_SMOKE_OK. Do not read or modify files.")
                items = [e["params"]["item"] for e in events if e.get("method") == "item/completed" and e["params"].get("threadId") == aid]
                report["astra"] = {"provider": astra["modelProvider"], "status": completed[aid]["status"],
                                   "command_completed": any(i["type"] == "commandExecution" and "BRIDGE_TOOL_OK" in (i.get("aggregatedOutput") or "") for i in items),
                                   "response_received": any(i["type"] == "agentMessage" and "BRIDGE_SMOKE_OK" in i.get("text", "") for i in items)}
                assert report["astra"]["command_completed"] and report["astra"]["response_received"], report["astra"]
        finally:
            if tid:
                try:
                    await core.call("thread/archive", {"threadId": tid})
                    report["fixture_archived"] = True
                except Exception:
                    report["fixture_archived"] = False
            await core.close()
            registry.close()
            server.shutdown()
            server.server_close()
            thread.join()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-astra", action="store_true", help="Send one real ChatGPT-subscription turn")
    options = parser.parse_args()
    print(json.dumps(asyncio.run(run(options.live_astra)), ensure_ascii=False, indent=2))
