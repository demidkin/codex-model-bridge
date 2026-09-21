"""CLI entry point; stdout is reserved for the app-server JSONL protocol."""

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from .config import Settings
from .registry import Registry
from .router import Router
from .rpc import CoreClient, JsonWriter, MAX_LINE, RpcError


async def serve(settings, arguments):
    """Run a multiplexed stdio session and clean up its children on disconnect."""
    settings.prepare_state()
    registry = Registry(settings.state / "routes.sqlite3")
    registry.recover()
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=MAX_LINE)
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin.buffer)
    transport, protocol = await loop.connect_write_pipe(
        asyncio.streams.FlowControlMixin, sys.stdout.buffer,
    )
    output = JsonWriter(asyncio.StreamWriter(transport, protocol, None, loop))
    router = Router(settings, registry, output.send)
    core = CoreClient([settings.core, *arguments], router.event)
    router.core = core
    if settings.enable_claude:
        from .claude import ClaudeEngine
        router.claude = ClaudeEngine(settings, registry, router.external_event, core)
        router.claude.agents = router.agents
        router.claude.skills = router.skills
    await core.start()
    marker = settings.state / "last-session.json"
    marker.write_text(json.dumps({
        "started_at": int(time.time()), "adapter_pid": os.getpid(),
        "core_pid": core.process.pid, "core": settings.core,
        "deepseek_enabled": settings.enable_deepseek,
        "claude_enabled": settings.enable_claude,
    }, indent=2) + "\n")
    requests = set()
    stopped = asyncio.Event()
    def on_signal(sig):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(core.process.pid, sig)
        stopped.set()

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        loop.add_signal_handler(sig, on_signal, sig)

    async def handle(message):
        try:
            result = await router.request(message["method"], message.get("params"))
            await output.send({"id": message["id"], "result": result})
        except RpcError as exc:
            await output.send({"id": message["id"], "error": exc.error})
        except (ValueError, ConnectionError) as exc:
            await output.send({"id": message["id"], "error": {"code": -32001, "message": str(exc)}})
        except Exception as exc:
            print(f"codex-bridge: request failed ({type(exc).__name__})", file=sys.stderr)
            await output.send({"id": message["id"], "error": {
                "code": -32603, "message": "Bridge internal error; no automatic retry was performed",
            }})

    async def receive():
        while raw := await reader.readline():
            message = json.loads(raw)
            if not isinstance(message, dict):
                raise ValueError("Expected a JSON-RPC object")
            if "method" in message and "id" in message:
                if len(requests) >= 2048:
                    raise ValueError("Too many pending host requests")
                task = asyncio.create_task(handle(message))
                requests.add(task)
                def collect(finished):
                    requests.discard(finished)
                    if not finished.cancelled() and finished.exception():
                        print("codex-bridge: host connection closed before response delivery", file=sys.stderr)
                task.add_done_callback(collect)
            elif router.claude and router.claude.accept_response(message):
                continue
            else:
                await core.send(message)

    input_task = asyncio.create_task(receive())
    signal_task = asyncio.create_task(stopped.wait())
    try:
        done, _ = await asyncio.wait(
            [input_task, core.reader_task, signal_task], return_when=asyncio.FIRST_COMPLETED,
        )
        if input_task in done:
            input_task.result()
            if requests:
                await asyncio.wait(requests, timeout=3)
        if core.reader_task in done:
            core.reader_task.result()
    finally:
        input_task.cancel()
        signal_task.cancel()
        await router.close()
        await core.close()
        for task in requests:
            task.cancel()
        await asyncio.gather(input_task, signal_task, *requests, return_exceptions=True)
        registry.close()
        transport.close()
    code = core.process.returncode or 0
    return 128 - code if code < 0 else code


def main():
    """Pass normal CLI operations through, intercepting only stdio app-server runs."""
    os.umask(0o077)
    settings = Settings.load()
    arguments = sys.argv[1:]
    if arguments and arguments[0] == "bridge-doctor":
        from .doctor import main as doctor_main
        return asyncio.run(doctor_main(settings, arguments[1:]))
    app_server = "app-server" in arguments
    utility = any(arg in ("generate-json-schema", "generate-ts", "--help", "-h", "--version", "-V") for arg in arguments)
    if not app_server or utility:
        os.execv(settings.core, [settings.core, *arguments])
    if settings.enable_deepseek or settings.enable_claude:
        version = subprocess.run(
            [settings.core, "--version"], capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip()
        if version != settings.tested_core_version:
            raise ValueError("Original Codex version changed; external routing requires a compatibility check")
    if settings.enable_claude:
        version = subprocess.run(
            [settings.claude, "--version"], capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip()
        if version != settings.tested_claude_version:
            raise ValueError("Claude Code version changed; its protocol requires a compatibility check")
    for index, argument in enumerate(arguments):
        if argument == "--listen" and index + 1 < len(arguments) and arguments[index + 1] != "stdio://":
            raise ValueError("The bridge currently supports app-server stdio only")
        if argument.startswith("--listen=") and argument != "--listen=stdio://":
            raise ValueError("The bridge currently supports app-server stdio only")
    return asyncio.run(serve(settings, arguments))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as error:
        print(f"codex-bridge: {error}", file=sys.stderr)
        sys.exit(1)
