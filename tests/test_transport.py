"""Exercise real pipes, out-of-order responses, host approvals, EOF, and exit."""

import asyncio
import sys
import unittest
from pathlib import Path
from bridge.rpc import CoreClient


class TransportTests(unittest.IsolatedAsyncioTestCase):
    """Validate that multiplexing does not confuse responses or approval IDs."""

    async def asyncSetUp(self):
        """Launch a small deterministic protocol peer."""
        self.events = []

        async def event(message):
            self.events.append(message)
            if "id" in message:
                await self.core.send({"id": message["id"], "result": {"decision": "decline"}})

        self.core = CoreClient([sys.executable, str(Path(__file__).with_name("fake_server.py"))], event)
        await self.core.start()

    async def asyncTearDown(self):
        """Collect every fixture process after each check."""
        await self.core.close()

    async def test_large_unicode_payload_and_unknown_fields(self):
        """Pass content exceeding asyncio's default line limit without truncation."""
        payload = {"futureField": {"text": "Привет 🦉" * 200000}}
        self.assertEqual(await self.core.call("echo", payload), payload)

    async def test_responses_out_of_order(self):
        """Correlate concurrent requests independently of completion order."""
        result = await asyncio.gather(self.core.call("reverse", {"n": 1}), self.core.call("reverse", {"n": 2}))
        self.assertEqual(result, [{"n": 1}, {"n": 2}])

    async def test_denial_reaches_native_approval_id(self):
        """Preserve server-initiated approval IDs and user decisions."""
        self.assertEqual(await self.core.call("approval", {}), {"decision": "decline"})
        self.assertEqual(self.events[0]["id"], 17)

    async def test_child_exit_releases_pending_request(self):
        """Surface process death without hanging or replaying the request."""
        with self.assertRaises(ConnectionError):
            await asyncio.wait_for(self.core.call("crash", {}), 3)
        await self.core.process.wait()
        self.assertEqual(self.core.process.returncode, 23)

    async def test_close_reaps_child(self):
        """Closing the bridge also closes its child process."""
        await self.core.close()
        self.assertIsNotNone(self.core.process.returncode)
