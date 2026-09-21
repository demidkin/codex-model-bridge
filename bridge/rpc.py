"""Asynchronous JSONL transport with request correlation and bounded shutdown."""

import asyncio
import contextlib
import json
import os
import signal


MAX_LINE = 64 * 1024 * 1024


class RpcError(Exception):
    """An explicit JSON-RPC failure that must reach the caller unchanged."""

    def __init__(self, error):
        super().__init__(error.get("message", "RPC error"))
        self.error = error


async def stop_process(process):
    """Close a child, then terminate its own process group if it does not exit."""
    if process.returncode is not None:
        return
    if process.stdin:
        process.stdin.close()
    try:
        await asyncio.wait_for(process.wait(), 3)
        return
    except TimeoutError:
        pass
    for sig in (signal.SIGTERM, signal.SIGKILL):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, sig)
        try:
            await asyncio.wait_for(process.wait(), 3)
            return
        except TimeoutError:
            pass


class JsonWriter:
    """Serialize complete JSON lines without interleaving concurrent writers."""

    def __init__(self, stream):
        self.stream = stream
        self.lock = asyncio.Lock()

    async def send(self, message):
        """Write one JSON object and respect backpressure."""
        data = (json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        if len(data) > MAX_LINE:
            raise ValueError("Protocol message exceeds the 64 MiB limit")
        async with self.lock:
            self.stream.write(data)
            await self.stream.drain()


class CoreClient:
    """Multiplex host requests to the original app-server without billing changes."""

    def __init__(self, command, event_handler):
        self.command = command
        self.event_handler = event_handler
        self.pending = {}
        self.sequence = 0
        self.process = None
        self.reader_task = None

    async def start(self):
        """Start the original binary in an isolated process group, inheriting stderr."""
        self.process = await asyncio.create_subprocess_exec(
            *self.command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            limit=MAX_LINE, start_new_session=True,
        )
        self.writer = JsonWriter(self.process.stdin)
        self.reader_task = asyncio.create_task(self._read())

    async def call(self, method, params):
        """Await exactly one correlated result; never retry requests automatically."""
        self.sequence += 1
        ident = f"bridge.core.{self.sequence}"
        future = asyncio.get_running_loop().create_future()
        self.pending[ident] = future
        try:
            await self.writer.send({"id": ident, "method": method, "params": params})
            # Some native methods deliberately long-poll; there is no global timeout.
            return await future
        finally:
            self.pending.pop(ident, None)

    async def send(self, message):
        """Pass notifications and native approval responses through unmodified."""
        await self.writer.send(message)

    async def close(self):
        """Stop the original process and collect its reader task."""
        if self.process:
            await stop_process(self.process)
        if self.reader_task:
            await asyncio.gather(self.reader_task, return_exceptions=True)

    async def _read(self):
        try:
            while raw := await self.process.stdout.readline():
                message = json.loads(raw)
                if "method" not in message and message.get("id") in self.pending:
                    future = self.pending[message["id"]]
                    if not future.done():
                        if "error" in message:
                            future.set_exception(RpcError(message["error"]))
                        else:
                            future.set_result(message.get("result"))
                else:
                    await self.event_handler(message)
        finally:
            for future in list(self.pending.values()):
                if not future.done():
                    future.set_exception(ConnectionError("Original app-server disconnected"))
