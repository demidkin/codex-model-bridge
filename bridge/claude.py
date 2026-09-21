"""Translate official Claude Code's stdio session into Codex task events.

The native CLI owns authentication, inference, tools and its conversation file.
The adapter owns only Codex UI history and explicit host approval replies.
"""
import asyncio
import base64
import copy
import contextlib
import difflib
import json
import mimetypes
import os
import re
import sys
import time
import uuid
from pathlib import Path
from .config import subscription_environment
from .leases import Lease, is_owned
from .doctor import command
from .rpc import JsonWriter, MAX_LINE, RpcError, stop_process

TOOLS = 'Read,Glob,Grep,Bash,Write,Edit,AskUserQuestion'
IMAGE_INPUT_TYPES = ('image', 'localImage')
# Mirrors Anthropic's own per-image request limit; a clearer failure than a
# provider-side rejection deep inside an already-running Claude process.
MAX_IMAGE_BYTES = 5 * 1024 * 1024
_IMAGE_SIGNATURES = (
    (b'\x89PNG\r\n\x1a\n', 'image/png'),
    (b'\xff\xd8\xff', 'image/jpeg'),
    (b'GIF87a', 'image/gif'),
    (b'GIF89a', 'image/gif'),
)

# Matches a single leading absolute path or URL token (with optional trailing
# punctuation) so raw pasted paths do not become the entire thread title.
_LEADING_PATH_RE = re.compile(
    r'^\s*(?:[a-zA-Z][a-zA-Z0-9+.-]*://\S+|/\S+|[A-Za-z]:\\\S+)[\s:,-]*'
)


def thread_title(text, limit=80):
    """First-message title, skipping a leading pasted path/URL if present."""
    stripped = _LEADING_PATH_RE.sub('', text, count=1).strip()
    return (stripped or text.strip())[:limit]


def _sniff_image_type(data):
    for signature, media_type in _IMAGE_SIGNATURES:
        if data.startswith(signature):
            return media_type
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'image/webp'
    return None


def _image_block(data, media_type):
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError('Image exceeds the 5 MB per-image limit for Claude tasks')
    if not media_type or not media_type.startswith('image/'):
        media_type = _sniff_image_type(data) or media_type
    if not media_type or not media_type.startswith('image/'):
        raise ValueError('Could not determine an image media type for this attachment')
    return {'type': 'image', 'source': {
        'type': 'base64', 'media_type': media_type, 'data': base64.b64encode(data).decode('ascii'),
    }}


def content_block(item):
    """Translate one Codex input item into an Anthropic Messages content block."""
    kind = item.get('type')
    if kind == 'text':
        return {'type': 'text', 'text': item['text']}
    if kind == 'localImage':
        path = item['path']
        data = Path(path).read_bytes()
        return _image_block(data, mimetypes.guess_type(path)[0])
    if kind == 'image':
        url = item.get('url')
        if url and url.startswith('data:'):
            header, separator, encoded = url.partition(',')
            if not separator or ';base64' not in header:
                raise ValueError('Only base64 data URLs are supported for pasted images')
            media_type = header[len('data:'):].split(';')[0]
            return _image_block(base64.b64decode(encoded), media_type)
        if url:
            return {'type': 'image', 'source': {'type': 'url', 'url': url}}
        raise ValueError('Claude tasks cannot resolve an uploaded image without a local path or URL')
    raise ValueError(f'Claude tasks do not support {kind!r} input yet')


def agents_md_context(cwd):
    """Read the same global and project AGENTS.md files native Codex loads for a turn.

    Claude Code has no equivalent of its own here (CLAUDE.md is disabled via
    --safe-mode), and its turns never reach native core's own instruction
    loading, so the bridge must read these files itself.
    """
    blocks = []
    home = Path(os.environ.get('CODEX_HOME') or (Path.home() / '.codex'))
    global_path = home / 'AGENTS.md'
    if global_path.is_file():
        blocks.append(('Global Codex instructions ({})'.format(global_path), global_path))
    project = []
    current = Path(cwd).resolve()
    for directory in (current, *current.parents):
        candidate = directory / 'AGENTS.md'
        if candidate.is_file() and candidate.resolve() != global_path.resolve():
            project.append(candidate)
    blocks += [('Project Codex instructions ({})'.format(path), path) for path in reversed(project)]
    sections = []
    for title, path in blocks:
        try:
            text = path.read_text().strip()
        except OSError:
            continue
        if text:
            sections.append(f'# {title}\n\n{text}')
    return '\n\n'.join(sections)


class ClaudeEngine:
    def __init__(self, settings, registry, emit, core):
        self.settings, self.registry, self.emit, self.core = settings, registry, emit, core
        self.active = {}
        self.approvals = {}
        self.list_pages = {}
        self.agents = None
        self.skills = None

    async def check_auth(self):
        code, raw = await command(self.settings.claude, 'auth', 'status', env=subscription_environment())
        try:
            info = json.loads(raw) if code == 0 else {}
        except ValueError:
            info = {}
        if not (info.get('loggedIn') and info.get('authMethod') == 'claude.ai' and
                info.get('apiProvider') == 'firstParty' and info.get('subscriptionType')):
            raise ValueError('Claude Code must be signed in to a Claude subscription; run claude auth login')

    def decorate(self, thread, include_turns=True):
        route = self.registry.get(thread['id'])
        if not route or route['engine'] != 'claude':
            return
        turns = self.registry.turns(thread['id'])
        thread.update(model=route['model'], modelProvider=route['provider'],
                      turns=copy.deepcopy(turns) if include_turns else [],
                      status={'type': 'active', 'activeFlags': []} if thread['id'] in self.active or is_owned(self.settings.state, 'claude:' + thread['id']) else {'type': 'idle'})
        if turns:
            thread['updatedAt'] = turns[-1].get('completedAt') or turns[-1]['startedAt']
            first = next((i for i in turns[0]['items'] if i['type'] == 'userMessage'), None)
            if first:
                thread['preview'] = ' '.join(x.get('text', '') for x in first['content'])[:200]

    # Provider-agnostic thread metadata that native Codex already tracks for
    # every thread it creates, Claude-routed or not (the desktop client polls
    # these on every thread it opens; native Codex simply has no record of the
    # attachments/goal/queue concept for a Claude turn, so an empty answer is
    # correct, not a stub). Rejecting them here used to abort the desktop's
    # bootstrap fetch for a Claude task before it ever requested turns/items,
    # which is what made history look empty right after a restart.
    READONLY_PASSTHROUGH = ('thread/attachment/list', 'thread/goal/get', 'thread/queue/list', 'thread/loaded/list')

    async def request(self, method, params, route):
        tid = route['thread_id']
        if method == 'turn/start':
            return await self.start(params, route)
        if method == 'thread/read':
            response = await self.core.call(method, {**params, 'includeTurns': False})
            self.decorate(response['thread'], params.get('includeTurns', False))
            return response
        if method in ('thread/turns/list', 'thread/items/list'):
            turns = self.registry.turns(tid)
            if method == 'thread/turns/list':
                data = turns
                default_direction = 'desc'
            else:
                data = [{'turnId': t['id'], 'item': i} for t in turns for i in t['items']
                        if params.get('turnId') in (None, t['id'])]
                default_direction = 'asc'
            if (params.get('sortDirection') or default_direction) == 'desc':
                data = list(reversed(data))
            start = int(params.get('cursor') or 0)
            end = start + max(1, min(params.get('limit') or 100, 1000))
            return {'data': copy.deepcopy(data[start:end]), 'nextCursor': str(end) if end < len(data) else None}
        if method in self.READONLY_PASSTHROUGH:
            return await self.core.call(method, params)
        if method == 'thread/settings/update':
            if tid in self.active or is_owned(self.settings.state, 'claude:' + tid):
                raise ValueError('Wait for the Claude turn to finish before changing settings')
            # Metadata updates cannot invoke Claude or native inference.
            return await self.core.call(method, params)
        if method in ('thread/archive', 'thread/unsubscribe', 'thread/unarchive'):
            if method == 'thread/archive' and (tid in self.active or is_owned(self.settings.state, 'claude:' + tid)):
                await self.interrupt(tid, None)
            response = await self.core.call(method, params)
            if method == 'thread/archive':
                # Core never sees a Claude turn (turn/start is deliberately
                # never forwarded to it, since its bridge_claude provider is
                # metadata-only and has no real endpoint), so it always reads
                # this thread's has_user_event as false and its own periodic
                # maintenance archives it as an abandoned draft regardless of
                # the real conversation our own registry holds. Only trust an
                # archive request for a thread that is genuinely empty; a
                # populated one must stay visible in our list no matter what
                # core's blind view of it decides.
                if not self.registry.turns(tid):
                    self.registry.set_archived(tid, True)
            elif method == 'thread/unarchive':
                self.registry.set_archived(tid, False)
            return response
        if method in ('thread/name/set', 'thread/metadata/update'):
            response = await self.core.call(method, params)
            metadata = await self.core.call('thread/read', {'threadId': tid, 'includeTurns': False})
            self.registry.save_thread(metadata['thread'])
            return response
        raise ValueError(f'{method} is not supported for Claude Code tasks yet')

    async def list_threads(self, params):
        """Merge native pagination with Claude holders excluded by native history lists."""
        params = copy.deepcopy(params)
        cursor = params.pop('cursor', None)
        limit = max(1, min(params.get('limit') or 100, 1000))
        if cursor and cursor.startswith('bridge.list.'):
            state = self.list_pages.pop(cursor, None)
            if state is None:
                raise ValueError('Claude task list cursor expired; refresh the task list')
            if params != state['params']:
                raise ValueError('Task list filters changed; refresh the task list')
            data, native_cursor, exhausted = state['data'], state['cursor'], state['exhausted']
        else:
            data = []
            for thread in self.registry.claude_threads(bool(params.get('archived'))):
                try:
                    self.decorate(thread, include_turns=False)
                    if not self.registry.turns(thread['id']):
                        continue
                    cwd_filter = params.get('cwd')
                    directories = [cwd_filter] if isinstance(cwd_filter, str) else cwd_filter
                    if directories and Path(thread['cwd']).resolve() not in [Path(p).resolve() for p in directories]:
                        continue
                    if params.get('modelProviders') and thread['modelProvider'] not in params['modelProviders']:
                        continue
                    if params.get('searchTerm') and params['searchTerm'].lower() not in (str(thread.get('name') or '') + thread['preview']).lower():
                        continue
                    if any(params.get(key) and params[key] != thread.get(key) for key in ('projectId', 'sectionId', 'parentThreadId', 'ancestorThreadId')):
                        continue
                    if params.get('originators') and thread.get('originator') not in params['originators']:
                        continue
                    source = thread.get('source')
                    if params.get('sourceKinds') and source not in params['sourceKinds']:
                        continue
                except (KeyError, TypeError, ValueError, OSError) as exc:
                    # One malformed or legacy registry row must not blank the
                    # whole merged list for every task sharing this request.
                    print(f"codex-bridge: skipped a malformed Claude thread entry ({type(exc).__name__})", file=sys.stderr)
                    continue
                data.append(thread)
            native_cursor, exhausted = cursor, False
        if not data and not (cursor or '').startswith('bridge.list.'):
            return await self.core.call('thread/list', {**params, 'cursor': cursor})
        fetched = 0
        while not exhausted and fetched < limit:
            response = await self.core.call('thread/list', {**params, 'cursor': native_cursor, 'limit': limit})
            next_native = response.get('nextCursor')
            if next_native and next_native == native_cursor:
                raise ValueError('Native task cursor did not advance')
            native_cursor = next_native
            exhausted = native_cursor is None
            for thread in response['data']:
                route = self.registry.get(thread['id'])
                if route and route['engine'] == 'claude':
                    continue
                data.append(thread)
                fetched += 1
        key = {'created_at':'createdAt','recency_at':'recencyAt','section_position':'sectionPosition'}.get(params.get('sortKey'), 'updatedAt')
        data.sort(key=lambda t: (t.get(key) or t.get('updatedAt') or 0, t['id']), reverse=params.get('sortDirection', 'desc') != 'asc')
        page, remaining = data[:limit], data[limit:]
        next_cursor = None
        if remaining or not exhausted:
            next_cursor = 'bridge.list.' + str(uuid.uuid4())
            self.list_pages[next_cursor] = {'params': params, 'data': remaining, 'cursor': native_cursor, 'exhausted': exhausted}
            while len(self.list_pages) > 256:
                self.list_pages.pop(next(iter(self.list_pages)))
        return {'data': page, 'nextCursor': next_cursor}

    async def start(self, params, route):
        tid = route['thread_id']
        if tid in self.active:
            raise ValueError('A Claude turn is already running in this task')
        lease = Lease(self.settings.state, 'claude:' + tid)
        if not lease.acquire():
            raise ValueError('This Claude task is active in another adapter process')
        try:
            content = params.get('input', [])
            if not content or any(item.get('type') not in ('text', *IMAGE_INPUT_TYPES) for item in content):
                raise ValueError('Claude tasks currently accept text and image input only')
            # Validate and translate before any subprocess exists, so a bad
            # attachment fails cleanly instead of surfacing mid-session.
            message_content = [content_block(item) for item in content]
            await self.check_auth()
            if not self.registry.turns(tid):
                # Force the native empty metadata holder to disk before inference.
                try:
                    await self.core.call('thread/read', {'threadId': tid, 'includeTurns': True})
                except RpcError as exc:
                    if exc.error.get('code') != -32601 or str(exc) != 'list_turns is not supported yet':
                        raise
                first_text = next((item['text'] for item in content if item.get('type') == 'text'), None)
                name = thread_title(first_text) if first_text else 'Image message'
                await self.core.call('thread/name/set', {'threadId': tid, 'name': name})
                # thread/name/set only updates native metadata; our own sidebar
                # listing is sourced entirely from the cached claude_threads row,
                # which would otherwise keep showing this task untitled until a
                # later resume refreshes it.
                metadata = await self.core.call('thread/read', {'threadId': tid, 'includeTurns': False})
                self.registry.save_thread(metadata['thread'])
            collaboration = params.get('collaborationMode') or {}
            if collaboration.get('mode') not in (None, 'default'):
                raise ValueError('Claude tasks currently support the default conversation mode')
            resume = bool(route.get('session_id'))
            route['session_id'] = route.get('session_id') or str(uuid.uuid4())
            for key in ('approvalPolicy', 'sandbox', 'cwd'):
                if key in params:
                    route['options'][key] = params[key]
            if params.get('cwd'):
                route['cwd'] = params['cwd']
            self.registry.save(route)
            turn = {'id': str(uuid.uuid4()), 'status': 'inProgress', 'items': [],
                    'startedAt': int(time.time()), 'completedAt': None, 'error': None}
            state = {'route': route, 'turn': turn, 'params': params, 'process': None,
                     'resume': resume, 'cancelled': False, 'tools': {}, 'text': {},
                     'controls': set(), 'lease': lease, 'content': copy.deepcopy(content),
                     'message_content': message_content}
            self.active[tid] = state
            self.registry.save_turn(tid, turn)
            state['task'] = asyncio.create_task(self.owned_run(state))
            return {'turn': copy.deepcopy(turn)}
        except BaseException:
            lease.close()
            raise

    async def event(self, state, method, **extra):
        await self.emit({'method': method, 'params': {
            'threadId': state['route']['thread_id'], 'turnId': state['turn']['id'], **extra}})

    async def owned_run(self, state):
        try:
            await self.run(state)
        finally:
            # Even a disconnected UI writer cannot leave session ownership locked.
            if state['process']:
                await stop_process(state['process'])
            self.active.pop(state['route']['thread_id'], None)
            state['lease'].close()

    async def add_item(self, state, item, completed=False):
        state['turn']['items'].append(item)
        self.registry.save_turn(state['route']['thread_id'], state['turn'])
        await self.event(state, 'item/started', item=copy.deepcopy(item))
        if completed:
            await self.event(state, 'item/completed', item=copy.deepcopy(item))

    async def run(self, state):
        route, turn = state['route'], state['turn']
        tid = route['thread_id']
        try:
            await self.event(state, 'turn/started', turn=copy.deepcopy(turn))
            await self.emit({'method': 'thread/status/changed', 'params': {
                'threadId': tid, 'status': {'type': 'active', 'activeFlags': []}}})
            await self.add_item(state, {'type': 'userMessage', 'id': str(uuid.uuid4()), 'content': state['content']}, True)
            policy = route['options'].get('approvalPolicy', 'untrusted')
            readonly = route['options'].get('sandbox') == 'read-only' or (state['params'].get('sandboxPolicy') or {}).get('type') == 'readOnly'
            permissions = {'ask': ['Bash', 'Write', 'Edit']}
            if readonly:
                permissions['deny'] = ['Bash', 'Write', 'Edit']
            argv = [self.settings.claude, '-p', '--input-format', 'stream-json', '--output-format', 'stream-json',
                    '--verbose', '--include-partial-messages', '--permission-prompt-tool', 'stdio',
                    '--permission-mode', 'manual', '--safe-mode', '--strict-mcp-config', '--tools', TOOLS,
                    '--settings', json.dumps({'permissions': permissions}), '--model', route['model'].removeprefix('claude-code/')]
            servers = {}
            if self.agents: servers['bridge_agents'] = {'type':'sdk','name':'bridge_agents'}
            if self.skills: servers['bridge_skills'] = {'type':'sdk','name':'bridge_skills'}
            if servers: argv += ['--mcp-config', json.dumps({'mcpServers':servers})]
            system_prompt = '\n\n'.join(filter(None, [
                agents_md_context(route['cwd']), state['params'].get('_codex_skill_context'),
            ]))
            if system_prompt:
                argv += ['--append-system-prompt', system_prompt]
            effort = state['params'].get('effort')
            if effort:
                argv += ['--effort', effort]
            argv += [('--resume=' if state['resume'] else '--session-id=') + route['session_id']]
            process = await asyncio.create_subprocess_exec(*argv, cwd=route['cwd'], env=subscription_environment(),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                limit=MAX_LINE, start_new_session=True)
            state['process'] = process
            state['writer'] = JsonWriter(process.stdin)
            await state['writer'].send({'type': 'control_request', 'request_id': 'bridge-init', 'request': {'subtype': 'initialize'}})
            async with asyncio.timeout(30):
                while raw := await process.stdout.readline():
                    msg = json.loads(raw)
                    if msg.get('type') == 'control_request':
                        await self.control(state, msg, policy)
                    if msg.get('type') == 'control_response' and msg['response']['request_id'] == 'bridge-init':
                        if msg['response']['subtype'] != 'success':
                            raise ValueError('Claude initialization failed')
                        break
                else:
                    raise ConnectionError('Claude exited during initialization')
            if state['cancelled']:
                raise asyncio.CancelledError
            # Persist session ownership before the only user dispatch. Never replay it.
            await state['writer'].send({'type': 'user', 'session_id': route['session_id'],
                'message': {'role': 'user', 'content': state['message_content']},
                'parent_tool_use_id': None})
            succeeded = False
            while raw := await process.stdout.readline():
                msg = json.loads(raw)
                kind = msg.get('type')
                if kind == 'control_request':
                    task = asyncio.create_task(self.control(state, msg, policy))
                    state['controls'].add(task)
                    task.add_done_callback(state['controls'].discard)
                elif kind == 'control_cancel_request':
                    for ident, approval in list(self.approvals.items()):
                        if approval['state'] is state and approval['request_id'] == msg.get('request_id'):
                            approval['future'].cancel()
                elif kind == 'stream_event':
                    await self.stream(state, msg.get('event') or {})
                elif kind == 'assistant':
                    await self.assistant(state, msg.get('message') or {})
                elif kind == 'user':
                    await self.tool_results(state, (msg.get('message') or {}).get('content', []))
                elif kind == 'result':
                    if msg.get('is_error') or msg.get('subtype') != 'success':
                        raise ValueError('Claude turn failed: ' + str(msg.get('subtype', 'error')))
                    succeeded = True
                    break
            if not succeeded:
                raise ConnectionError('Claude disconnected before completing the turn; no retry performed')
            turn['status'] = 'interrupted' if state['cancelled'] else 'completed'
        except asyncio.CancelledError:
            turn['status'] = 'interrupted'
        except Exception as exc:
            turn['status'] = 'interrupted' if state['cancelled'] else 'failed'
            if not state['cancelled']:
                turn['error'] = {'message': str(exc), 'codexErrorInfo': None, 'additionalDetails': None}
                await self.event(state, 'error', error=turn['error'], willRetry=False)
        finally:
            for ident, approval in list(self.approvals.items()):
                if approval['state'] is state and not approval['future'].done():
                    approval['future'].cancel()
            for task in list(state['controls']):
                task.cancel()
            await asyncio.gather(*state['controls'], return_exceptions=True)
            if state['process']:
                await stop_process(state['process'])
            for item in turn['items']:
                if item.get('status') == 'inProgress':
                    item['status'] = 'failed'
                    await self.event(state, 'item/completed', item=copy.deepcopy(item))
            turn['completedAt'] = int(time.time())
            turn['durationMs'] = max(0, (turn['completedAt'] - turn['startedAt']) * 1000)
            self.registry.save_turn(tid, turn)
            self.active.pop(tid, None)
            await self.event(state, 'turn/completed', turn=copy.deepcopy(turn))
            await self.emit({'method': 'thread/status/changed', 'params': {'threadId': tid, 'status': {'type': 'idle'}}})

    async def stream(self, state, event):
        kind = event.get('type')
        if kind == 'message_start':
            state['message_id'] = event['message']['id']
        elif kind == 'content_block_start' and event['content_block'].get('type') == 'text':
            ident = state.get('message_id', str(uuid.uuid4())) + ':' + str(event['index'])
            item = {'type': 'agentMessage', 'id': ident, 'text': '', 'phase': None}
            state['text'][ident] = item
            await self.add_item(state, item)
        elif kind == 'content_block_delta' and event['delta'].get('type') == 'text_delta':
            ident = state.get('message_id', '') + ':' + str(event['index'])
            if ident in state['text']:
                delta = event['delta']['text']
                state['text'][ident]['text'] += delta
                await self.event(state, 'item/agentMessage/delta', itemId=ident, delta=delta)

    async def assistant(self, state, message):
        for index, block in enumerate(message.get('content', [])):
            if block.get('type') == 'text':
                ident = message.get('id', str(uuid.uuid4())) + ':' + str(index)
                item = state['text'].get(ident)
                if item is None:
                    # Claude periodically repeats an already-shown message verbatim as
                    # an extra content block that was never streamed (so it has no
                    # prior ident here); only skip a brand-new block whose text exactly
                    # matches one already rendered in this turn, never a genuinely new one.
                    if block['text'] and any(
                        existing.get('type') == 'agentMessage' and existing.get('text') == block['text']
                        for existing in state['turn']['items']
                    ):
                        continue
                    item = {'type': 'agentMessage', 'id': ident, 'text': block['text'], 'phase': None}
                    await self.add_item(state, item)
                item['text'] = block['text']
                await self.event(state, 'item/completed', item=copy.deepcopy(item))
            elif block.get('type') == 'tool_use':
                await self.ensure_tool(state, block['id'], block['name'], block.get('input') or {})

    async def ensure_tool(self, state, ident, name, args):
        if ident in state['tools']:
            return state['tools'][ident]
        if name == 'Bash':
            item = {'type': 'commandExecution', 'id': ident, 'command': args.get('command', ''),
                    'cwd': state['route']['cwd'], 'status': 'inProgress', 'commandActions': [],
                    'aggregatedOutput': '', 'exitCode': None, 'durationMs': None, 'source': 'agent'}
        elif name in ('Write', 'Edit'):
            path = str(Path(state['route']['cwd']) / args.get('file_path', ''))
            target = Path(path)
            exists = target.exists()
            if exists and (not target.is_file() or target.stat().st_size > 2 * 1024 * 1024):
                raise ValueError('Claude file approval supports regular text files up to 2 MiB')
            old = target.read_text() if exists else ''
            if name == 'Write':
                new = args.get('content', '')
            else:
                before = args.get('old_string', '')
                if not before or before not in old:
                    raise ValueError('Claude edit no longer matches the current file')
                new = old.replace(before, args.get('new_string', ''), -1 if args.get('replace_all') else 1)
            diff = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile=path, tofile=path))
            item = {'type': 'fileChange', 'id': ident, 'status': 'inProgress', 'changes': [
                {'path': path, 'diff': diff, 'kind': {'type': 'update', 'move_path': None} if exists else {'type': 'add'}}]}
        else:
            item = {'type': 'dynamicToolCall', 'id': ident, 'tool': name, 'arguments': args,
                    'namespace': 'claude', 'status': 'inProgress', 'contentItems': None, 'success': None, 'durationMs': None}
        state['tools'][ident] = item
        await self.add_item(state, item)
        return item

    async def tool_results(self, state, content):
        if not isinstance(content, list):
            return
        for block in content:
            if block.get('type') != 'tool_result':
                continue
            item = state['tools'].get(block.get('tool_use_id'))
            if not item:
                continue
            body = block.get('content', '')
            text = body if isinstance(body, str) else '\n'.join(x.get('text', '') for x in body if isinstance(x, dict))
            failed = bool(block.get('is_error'))
            if item.get('status') != 'declined':
                item['status'] = 'failed' if failed else 'completed'
            if item['type'] == 'commandExecution':
                item['aggregatedOutput'] = text
                # Claude tool_result does not expose an authoritative shell exit code.
            elif item['type'] == 'dynamicToolCall':
                item.update(contentItems=[{'type': 'inputText', 'text': text}], success=not failed)
            self.registry.save_turn(state['route']['thread_id'], state['turn'])
            await self.event(state, 'item/completed', item=copy.deepcopy(item))

    async def control(self, state, msg, policy):
        ident, request = msg['request_id'], msg['request']
        response = {'behavior': 'deny', 'message': 'Unsupported host request', 'interrupt': True}
        try:
            if request.get('subtype') == 'mcp_message':
                service = {'bridge_agents':self.agents, 'bridge_skills':self.skills}.get(request.get('server_name'))
                if not service: raise ValueError('Unknown SDK MCP server')
                response = {'mcp_response': await service.mcp(state['route']['thread_id'], state['turn']['id'], request['message'])}
            elif request.get('subtype') == 'can_use_tool':
                name, args = request['tool_name'], request.get('input') or {}
                tool_id = request.get('tool_use_id') or str(uuid.uuid4())
                item = await self.ensure_tool(state, tool_id, name, args)
                allowed = False
                route_options = state['route']['options']
                # Full access can be represented three different ways
                # depending on how the task's policy reached us: a named
                # permission profile, a plain sandbox string, or a full
                # sandboxPolicy object. Checking only the last form left
                # Claude re-gating Bash/Write/Edit behind a host approval
                # prompt for tasks that were actually on danger-full-access.
                full_access = (
                    route_options.get('permissions') == ':danger-full-access'
                    or route_options.get('sandbox') == 'danger-full-access'
                    or (route_options.get('sandboxPolicy') or {}).get('type') == 'dangerFullAccess'
                )
                if name in ('Bash', 'Write', 'Edit') and full_access and not args.get('run_in_background'):
                    # Codex's own danger-full-access sandbox already grants unrestricted
                    # execution; Claude's tool calls must not be re-gated behind a host
                    # approval request that this mode never intends to show.
                    allowed = True
                elif name in ('Bash', 'Write', 'Edit') and policy != 'never' and not args.get('run_in_background'):
                    method = 'item/commandExecution/requestApproval' if name == 'Bash' else 'item/fileChange/requestApproval'
                    params = {'threadId': state['route']['thread_id'], 'turnId': state['turn']['id'],
                              'itemId': tool_id, 'startedAtMs': int(time.time()*1000),
                              'reason': request.get('decision_reason') or 'Claude Code requests permission'}
                    if name == 'Bash':
                        params.update(command=args.get('command', ''), cwd=state['route']['cwd'],
                                      availableDecisions=['accept', 'decline', 'cancel'])
                    reply = await self.ask_host(state, ident, method, params)
                    allowed = reply.get('decision') in ('accept', 'acceptForSession')
                elif self.agents and name.startswith('mcp__bridge_agents__') and name.removeprefix('mcp__bridge_agents__') in {s['name'] for s in self.agents.specs}:
                    allowed = True  # Child tool actions still use inherited host approvals.
                elif self.skills and name.startswith('mcp__bridge_skills__') and name.removeprefix('mcp__bridge_skills__') in {s['name'] for s in self.skills.specs}:
                    allowed = True  # Reading skills never approves their commands; helper inherits policy.
                elif name == 'AskUserQuestion':
                    questions = [{'id': str(index), 'header': q.get('header', 'Claude')[:12], 'question': q['question'],
                        'options': q.get('options'), 'isOther': True, 'isSecret': False} for index, q in enumerate(args.get('questions', []))]
                    reply = await self.ask_host(state, ident, 'item/tool/requestUserInput', {
                        'threadId': state['route']['thread_id'], 'turnId': state['turn']['id'],
                        'itemId': tool_id, 'isBlocking': True, 'questions': questions})
                    answers = reply.get('answers', {})
                    if answers:
                        args = {**args, 'answers': {q['question']: ', '.join(answers.get(str(i), {}).get('answers', []))
                                                   for i, q in enumerate(args.get('questions', []))}}
                        allowed = True
                if allowed and not state['cancelled']:
                    response = {'behavior': 'allow', 'updatedInput': args}
                else:
                    item['status'] = 'declined' if item['type'] in ('commandExecution', 'fileChange') else 'failed'
                    response = {'behavior': 'deny', 'message': 'Not approved by the Codex host', 'interrupt': False}
            await state['writer'].send({'type': 'control_response', 'response': {
                'subtype': 'success', 'request_id': ident, 'response': response}})
        except asyncio.CancelledError:
            return
        except Exception:
            # A failed approval translation must never allow the operation.
            await state['writer'].send({'type': 'control_response', 'response': {
                'subtype': 'success', 'request_id': ident, 'response': {
                    'behavior': 'deny', 'message': 'Host approval failed', 'interrupt': True}}})

    async def ask_host(self, state, request_id, method, params):
        ident = 'bridge.claude.' + str(uuid.uuid4())
        future = asyncio.get_running_loop().create_future()
        self.approvals[ident] = {'future': future, 'state': state, 'request_id': request_id}
        try:
            await self.emit({'id': ident, 'method': method, 'params': params})
            return await future
        finally:
            self.approvals.pop(ident, None)

    def accept_response(self, message):
        ident = message.get('id')
        if 'method' in message or not isinstance(ident, str) or not ident.startswith('bridge.claude.'):
            return False
        entry = self.approvals.get(ident)
        if entry and not entry['future'].done():
            entry['future'].set_result(message.get('result') or {})
        return True

    async def interrupt(self, thread_id, turn_id):
        state = self.active.get(thread_id)
        if not state:
            if is_owned(self.settings.state, 'claude:' + thread_id):
                raise ValueError('This Claude task is active in another adapter process; stop it in its owning task')
            return {}
        if turn_id and state['turn']['id'] != turn_id:
            raise ValueError('Claude turn id does not match the active turn')
        state['cancelled'] = True
        await asyncio.sleep(0)  # Let a newly scheduled turn enter its cleanup scope.
        if state.get('writer'):
            with contextlib.suppress(ConnectionError, OSError):
                await state['writer'].send({'type':'control_request', 'request_id':'bridge-interrupt-' + str(uuid.uuid4()),
                                            'request':{'subtype':'interrupt'}})
        done, _ = await asyncio.wait([state['task']], timeout=2)
        if not done:
            state['task'].cancel()
        await asyncio.gather(state['task'], return_exceptions=True)
        return {}

    async def close(self):
        await asyncio.gather(*(self.interrupt(tid, None) for tid in list(self.active)))
