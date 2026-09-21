"""Offline checks for approvals, isolation, history and sidebar pagination."""
import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
import base64
from bridge.claude import ClaudeEngine, content_block
from bridge.config import Settings, subscription_environment
from bridge.registry import Registry
from test_routes import FakeCore

PNG_BYTES = base64.b64decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII='
)

class CaptureWriter:
    def __init__(self):
        self.messages = []
    async def send(self, message):
        self.messages.append(copy.deepcopy(message))

class ClaudeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = Settings(state=self.root, enable_claude=True)
        self.registry = Registry(self.root / 'routes.sqlite3')
        self.core = FakeCore()
        self.events = []
        async def emit(event):
            self.events.append(event)
        self.engine = ClaudeEngine(self.settings, self.registry, emit, self.core)
        self.route = {'thread_id': 'fixture', 'engine': 'claude', 'provider': 'bridge_claude',
                      'model': 'claude-code/sonnet', 'cwd': str(self.root), 'options': {}, 'locked': True}
        self.registry.save(self.route)
        self.state = {'route': self.route, 'turn': {'id': 'turn', 'items': [], 'status': 'inProgress', 'startedAt': 10},
                      'tools': {}, 'text': {}, 'writer': CaptureWriter(), 'cancelled': False}
    async def asyncTearDown(self):
        self.registry.close()
        self.temp.cleanup()
    async def test_approval_has_to_match_own_id(self):
        msg = {'request_id': 'native-1', 'request': {'subtype': 'can_use_tool', 'tool_name': 'Bash',
               'tool_use_id': 'tool', 'input': {'command': 'touch denied.txt'}}}
        task = asyncio.create_task(self.engine.control(self.state, msg, 'untrusted'))
        await asyncio.sleep(0)
        approval = self.events[-1]
        self.assertFalse(self.engine.accept_response({'id': 42, 'result': {'decision': 'accept'}}))
        self.assertFalse(task.done())
        self.engine.accept_response({'id': approval['id'], 'result': {'decision': 'decline'}})
        await task
        self.assertEqual(self.state['writer'].messages[0]['response']['response']['behavior'], 'deny')
        self.assertEqual(self.state['tools']['tool']['status'], 'declined')
    async def test_never_and_background_fail_closed(self):
        for policy, args in [('never', {'command': 'printf x'}), ('untrusted', {'command': 'sleep 2', 'run_in_background': True})]:
            self.state['writer'].messages.clear()
            await self.engine.control(self.state, {'request_id': 'request', 'request': {
                'subtype': 'can_use_tool', 'tool_name': 'Bash', 'input': args}}, policy)
            self.assertEqual(self.state['writer'].messages[0]['response']['response']['behavior'], 'deny')
            self.assertFalse(any('id' in event for event in self.events))
    async def test_unknown_method_cannot_reach_inference(self):
        with self.assertRaises(ValueError):
            await self.engine.request('review/start', {'threadId': 'fixture'}, self.route)
        self.assertEqual(self.core.calls, [])
    async def test_archive_is_ignored_for_a_thread_with_real_history(self):
        """Core never sees a Claude turn (turn/start is deliberately never
        forwarded to it), so its own periodic maintenance treats every Claude
        thread as an abandoned draft and archives it regardless of the real
        conversation this registry holds; only a genuinely empty thread
        should actually end up archived."""
        self.registry.save_thread({'id': 'fixture', 'cwd': str(self.root), 'modelProvider': 'bridge_claude', 'preview': ''})
        self.registry.save_turn('fixture', {'id': 'turn', 'items': [], 'status': 'completed', 'startedAt': 1, 'completedAt': 1})
        await self.engine.request('thread/archive', {'threadId': 'fixture'}, self.route)
        self.assertIn('fixture', [t['id'] for t in self.registry.claude_threads(archived=False)])
        self.assertNotIn('fixture', [t['id'] for t in self.registry.claude_threads(archived=True)])
    async def test_archive_is_honored_for_a_genuinely_empty_thread(self):
        self.registry.save_thread({'id': 'fixture', 'cwd': str(self.root), 'modelProvider': 'bridge_claude', 'preview': ''})
        await self.engine.request('thread/archive', {'threadId': 'fixture'}, self.route)
        self.assertIn('fixture', [t['id'] for t in self.registry.claude_threads(archived=True)])
    async def test_unarchive_always_restores_visibility(self):
        self.registry.save_thread({'id': 'fixture', 'cwd': str(self.root), 'modelProvider': 'bridge_claude', 'preview': ''})
        self.registry.set_archived('fixture', True)
        await self.engine.request('thread/unarchive', {'threadId': 'fixture'}, self.route)
        self.assertIn('fixture', [t['id'] for t in self.registry.claude_threads(archived=False)])
    async def test_readonly_metadata_methods_pass_through_to_native_core(self):
        """These must not fail-closed: the desktop bootstraps every open thread
        with them, and an error here used to abort loading turns entirely."""
        for method in ('thread/attachment/list', 'thread/goal/get', 'thread/queue/list', 'thread/loaded/list'):
            self.core.calls.clear()
            result = await self.engine.request(method, {'threadId': 'fixture'}, self.route)
            self.assertEqual(self.core.calls, [(method, {'threadId': 'fixture'})])
            self.assertEqual(result, {})
    async def test_items_list_defaults_ascending_turns_list_defaults_descending(self):
        for a, b in (('a', 10), ('b', 20)):
            self.registry.save_turn('fixture', {'id': a, 'status': 'completed', 'startedAt': b,
                                                 'items': [{'type': 'agentMessage', 'id': a, 'text': a}]})
        turns = await self.engine.request('thread/turns/list', {'threadId': 'fixture'}, self.route)
        self.assertEqual([t['id'] for t in turns['data']], ['b', 'a'])
        items = await self.engine.request('thread/items/list', {'threadId': 'fixture'}, self.route)
        self.assertEqual([i['item']['id'] for i in items['data']], ['a', 'b'])
    async def test_streamed_text_is_not_duplicated(self):
        await self.engine.stream(self.state, {'type':'message_start', 'message':{'id':'msg'}})
        await self.engine.stream(self.state, {'type':'content_block_start', 'index':0, 'content_block':{'type':'text'}})
        await self.engine.stream(self.state, {'type':'content_block_delta', 'index':0, 'delta':{'type':'text_delta','text':'Hello'}})
        await self.engine.assistant(self.state, {'id':'msg', 'content':[{'type':'text','text':'Hello'}]})
        self.assertEqual(len(self.state['turn']['items']), 1)
        self.assertEqual(self.state['turn']['items'][0]['text'], 'Hello')
    async def test_repeated_assistant_content_block_is_not_duplicated(self):
        """Reproduces a real live turn: Claude's non-streaming 'assistant' event
        carried the already-shown text again as an extra, never-streamed block."""
        await self.engine.stream(self.state, {'type':'message_start', 'message':{'id':'msg1'}})
        await self.engine.stream(self.state, {'type':'content_block_start', 'index':0, 'content_block':{'type':'text'}})
        await self.engine.stream(self.state, {'type':'content_block_delta', 'index':0, 'delta':{'type':'text_delta','text':'В чём была причина'}})
        await self.engine.assistant(self.state, {'id':'msg1', 'content':[
            {'type':'text','text':'В чём была причина'},
            {'type':'text','text':'В чём была причина'},
        ]})
        agent_messages = [i for i in self.state['turn']['items'] if i['type']=='agentMessage']
        self.assertEqual(len(agent_messages), 1)
        self.assertEqual(agent_messages[0]['text'], 'В чём была причина')

    async def test_list_merges_sorted_pages_and_keeps_filters(self):
        self.registry.save_thread({'id':'fixture','cwd':str(self.root),'updatedAt':25,'createdAt':25,
                                   'modelProvider':'bridge_claude','source':'cli','preview':''})
        self.registry.save_turn('fixture', {'id':'turn', 'items':[], 'status':'completed', 'startedAt':25,'completedAt':25})
        pages = {None: ([{'id':'native30','updatedAt':30},{'id':'native20','updatedAt':20}], 'next'),
                 'next': ([{'id':'native10','updatedAt':10}], None)}
        async def call(method, params):
            data, cursor = pages[params.get('cursor')]
            return {'data':copy.deepcopy(data), 'nextCursor':cursor}
        self.core.call = call
        first = await self.engine.list_threads({'limit':2})
        self.assertEqual([t['id'] for t in first['data']], ['native30','fixture'])
        second = await self.engine.list_threads({'limit':2,'cursor':first['nextCursor']})
        self.assertEqual([t['id'] for t in second['data']], ['native20','native10'])
        self.assertIsNone(second['nextCursor'])
        self.registry.set_archived('fixture', True)
        other = await self.engine.list_threads({'limit':2})
        self.assertNotIn('fixture', [t['id'] for t in other['data']])
    def test_content_block_reads_local_image_by_extension(self):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as handle:
            handle.write(PNG_BYTES)
            path = handle.name
        block = content_block({'type': 'localImage', 'path': path})
        self.assertEqual(block['source']['media_type'], 'image/png')
        self.assertEqual(base64.b64decode(block['source']['data']), PNG_BYTES)

    def test_content_block_sniffs_local_image_without_extension(self):
        with tempfile.NamedTemporaryFile(suffix='', delete=False) as handle:
            handle.write(PNG_BYTES)
            path = handle.name
        block = content_block({'type': 'localImage', 'path': path})
        self.assertEqual(block['source']['media_type'], 'image/png')

    def test_content_block_decodes_pasted_data_url(self):
        url = 'data:image/png;base64,' + base64.b64encode(PNG_BYTES).decode('ascii')
        block = content_block({'type': 'image', 'url': url})
        self.assertEqual(block['source']['media_type'], 'image/png')
        self.assertEqual(base64.b64decode(block['source']['data']), PNG_BYTES)

    def test_content_block_passes_through_remote_url(self):
        block = content_block({'type': 'image', 'url': 'https://example.com/a.png'})
        self.assertEqual(block, {'type': 'image', 'source': {'type': 'url', 'url': 'https://example.com/a.png'}})

    def test_content_block_rejects_unresolvable_image_reference(self):
        with self.assertRaises(ValueError):
            content_block({'type': 'image', 'fileId': 'file-123'})

    def test_content_block_rejects_unknown_type(self):
        with self.assertRaises(ValueError):
            content_block({'type': 'mention', 'path': '/tmp/x', 'name': 'x'})

    async def test_start_accepts_image_and_falls_back_title(self):
        """An image-only first message must not crash naming or dispatch."""
        from unittest.mock import AsyncMock
        self.engine.check_auth = AsyncMock(return_value=None)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as handle:
            handle.write(PNG_BYTES)
            path = handle.name
        renamed = {}
        async def call(method, params):
            self.core.calls.append((method, params))
            if method == 'thread/name/set':
                renamed['name'] = params['name']
            if method == 'thread/read':
                return {'thread': {'id': 'fixture', 'cwd': str(self.root),
                                    'modelProvider': 'bridge_claude', 'source': 'cli',
                                    'name': renamed.get('name'), 'preview': ''}}
            return {}
        self.core.call = call
        await self.engine.start({'input': [{'type': 'localImage', 'path': path}]}, self.route)
        state = self.engine.active['fixture']
        task = state['task']
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(renamed['name'], 'Image message')
        self.assertEqual(state['message_content'][0]['type'], 'image')

    async def test_start_refreshes_cached_title_for_new_task(self):
        """A brand-new task's auto-title must reach the sidebar's own cache too."""
        from unittest.mock import AsyncMock
        self.engine.check_auth = AsyncMock(return_value=None)
        renamed = {}
        async def call(method, params):
            self.core.calls.append((method, params))
            if method == 'thread/name/set':
                renamed['name'] = params['name']
            if method == 'thread/read':
                return {'thread': {'id': 'fixture', 'cwd': str(self.root),
                                    'modelProvider': 'bridge_claude', 'source': 'cli',
                                    'name': renamed.get('name'), 'preview': ''}}
            return {}
        self.core.call = call
        result = await self.engine.start({'input': [{'type': 'text', 'text': 'explain this file'}]}, self.route)
        task = self.engine.active['fixture']['task']
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(renamed['name'], 'explain this file')
        cached = self.registry.claude_threads()
        self.assertEqual(cached[0]['name'], 'explain this file')

    async def test_subscription_environment_removes_api_routes(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY':'fixture','CLAUDE_CODE_OAUTH_TOKEN':'fixture','CLAUDECODE':'1'}):
            filtered = subscription_environment()
            self.assertNotIn('ANTHROPIC_API_KEY',filtered)
            self.assertNotIn('CLAUDE_CODE_OAUTH_TOKEN',filtered)
            self.assertNotIn('CLAUDECODE',filtered)
            self.assertEqual(os.environ['ANTHROPIC_API_KEY'],'fixture')

    async def test_unsubscribe_does_not_stop_background_task(self):
        from unittest.mock import AsyncMock
        self.engine.active["fixture"] = self.state
        self.engine.interrupt = AsyncMock()
        await self.engine.request("thread/unsubscribe", {"threadId":"fixture"}, self.route)
        self.engine.interrupt.assert_not_awaited()
        self.assertIn("fixture", self.engine.active)
        self.assertEqual(self.core.calls[-1][0], "thread/unsubscribe")
