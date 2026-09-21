"""Model-selectable child tasks, owned by the adapter rather than a provider.

No inference retry, subscription token access, or implicit provider fallback.
Native dynamic tools and Claude's SDK MCP transport share this coordinator.
"""
import asyncio
import contextlib
import copy
import json
import time
import uuid

from .leases import Lease, is_owned
from .catalog import PROVIDERS


NAMESPACE = 'bridge_agents'
ACTIVE = {'starting', 'running'}
INSTRUCTIONS = (
    'When delegating with an explicitly chosen model, use bridge_agents tools. '
    'They support OpenAI/ChatGPT, DeepSeek API and official Claude Code subscription. '
    'Omitting model inherits the current task model; never default all children to DeepSeek. '
    'For an external model do not use the native OpenAI-only model override. '
    'Pass a self-contained task; the full parent conversation is not copied. '
    'Wait for results with wait_agents; a timeout means still running. '
    'Only delegate when the user or applicable instructions authorize delegation.'
)


def tool_specs(models):
    def tool(name, description, properties, required=()):
        return {'type': 'function', 'name': name, 'description': description,
                'inputSchema': {'type': 'object', 'properties': properties,
                                'required': list(required), 'additionalProperties': False}}
    string = {'type': 'string'}
    message = {'type': 'string', 'minLength': 1, 'maxLength': 64000}
    return [
        tool('spawn_agent', 'Start a child task with a chosen model. Omit model to inherit the parent model. '
             'No provider fallback. Native OpenAI models (e.g. gpt-6-astra) plus: ' + ', '.join(models) + '. '
             'Returns an agent id; use wait_agents for the result. ' + INSTRUCTIONS,
             {'message': message, 'model': string, 'reasoning_effort': string, 'task_name': string}, ('message',)),
        tool('wait_agents', 'Wait up to 60 seconds for children; return current status, model and result. '
             'If still running, wait again. This does not cancel work.',
             {'targets': {'type': 'array', 'items': string, 'minItems': 1, 'maxItems': 8},
              'timeout_ms': {'type': 'integer', 'minimum': 0, 'maximum': 60000}}, ('targets',)),
        tool('send_message', 'Continue a finished child using its existing session and model. '
             'Running or uncertain children cannot receive a second turn.',
             {'target': string, 'message': message}, ('target', 'message')),
        tool('interrupt_agent', 'Stop one child and its descendants, leaving siblings and parent running.',
             {'target': string}, ('target',)),
        tool('list_agents', 'List this parent task\'s children, including their selected model and state.', {}),
        tool('list_models', 'List exact model identifiers available for spawn_agent.', {}),
    ]


class AgentManager:
    def __init__(self, router):
        self.router, self.registry = router, router.registry
        self.state = router.settings.state
        self.specs = tool_specs(router.catalog.engines)
        self.leases = {}
        self.callbacks = set()
        self.locks = {}
        self.models = None
        self.stopping = False

    def inject(self, params):
        """Only add our namespace; preserve host tools and caller instructions."""
        params = copy.deepcopy(params)
        existing = params.get('dynamicTools') or []
        if any(t.get('name') == NAMESPACE for t in existing):
            raise ValueError('The bridge_agents dynamic tool namespace is reserved')
        params['dynamicTools'] = [*existing, {'type': 'namespace', 'name': NAMESPACE,
                                             'description': INSTRUCTIONS, 'tools': self.specs}]
        params['developerInstructions'] = '\n\n'.join(filter(None, [params.get('developerInstructions'), INSTRUCTIONS]))
        return params

    def background(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.callbacks.add(task)
        def collect(finished):
            self.callbacks.discard(finished)
            if not finished.cancelled():
                finished.exception()  # A disconnected host must not leave an unhandled task.
        task.add_done_callback(collect)
        return task

    def intercept(self, message):
        params = message.get('params') or {}
        if message.get('method') != 'item/tool/call' or params.get('namespace') != NAMESPACE:
            return False
        # CoreClient's reader must keep reading responses while the tool calls core.
        self.background(self.dynamic_call(message))
        return True

    async def dynamic_call(self, message):
        p = message['params']
        try:
            result = await self.call(p['threadId'], p['tool'], p.get('arguments'),
                                     str(p['turnId']) + ':' + str(p['callId']))
            reply = {'success': True, 'contentItems': [{'type': 'inputText', 'text': json.dumps(result, ensure_ascii=False)}]}
        except Exception as exc:
            reply = {'success': False, 'contentItems': [{'type': 'inputText', 'text': str(exc)}]}
        await self.router.core.send({'id': message['id'], 'result': reply})

    def observe(self, message):
        p = message.get('params') or {}
        tid = p.get('threadId') or (p.get('thread') or {}).get('id')
        agent = self.registry.agent(tid) if tid else None
        if agent and tid in self.leases:
            method = message.get('method')
            if method == 'turn/started':
                agent.update(turn_id=p['turn']['id'], status='running')
            elif method == 'turn/completed':
                turn = p['turn']
                if agent.get('turn_id') not in (None, turn['id']):
                    return
                agent.update(turn_id=turn['id'], status=turn['status'], error=turn.get('error'))
                texts = [i.get('text', '') for i in turn.get('items', []) if i.get('type') == 'agentMessage']
                if texts:
                    agent['result'] = '\n\n'.join(texts)[-256000:]
            elif method == 'item/completed' and p.get('item', {}).get('type') == 'agentMessage':
                # Some native completion notifications omit their items.
                agent['result'] = (agent.get('result', '') + '\n\n' + p['item'].get('text', '')).strip()[-256000:]
            elif method == 'item/completed' and p.get('item', {}).get('type') == 'imageGeneration':
                item = p['item']
                agent.setdefault('images', []).append({k:item[k] for k in ('id','status','savedPath','failure') if k in item})
            else:
                return
            agent['updated_at'] = int(time.time())
            self.registry.save_agent(agent)
            if agent['status'] not in ACTIVE:
                self.leases.pop(tid).close()
        if message.get('method') == 'turn/completed' and p['turn']['status'] in ('interrupted', 'failed'):
            self.background(self.cancel_children(tid, p['turn']['id']))

    def owned(self, parent, ident):
        agent = self.registry.agent(ident)
        if not agent or agent['parent_id'] != parent:
            raise ValueError('This agent does not belong to the requesting parent task')
        if agent['status'] in ACTIVE and ident not in self.leases and not is_owned(self.state, 'agent:' + ident):
            # A crashed owner's native process might have dispatched work. Never replay it.
            agent.update(status='uncertain', error={'message': 'Owner disconnected; execution outcome is unknown. No automatic retry.'})
            self.registry.save_agent(agent)
        return agent

    @staticmethod
    def public(agent):
        return {k: agent.get(k) for k in ('id', 'parent_id', 'task_name', 'model', 'engine',
                                         'reasoning_effort', 'status', 'turn_id', 'result', 'error')}

    async def available_models(self):
        if self.models is None:
            models, cursor = {}, None
            while True:
                reply = await self.router.core.call('model/list', {'includeHidden': False, 'limit': 100, 'cursor': cursor})
                models.update({m['model']: m for m in reply['data']})
                cursor = reply.get('nextCursor')
                if not cursor:
                    break
            models.update({m['model']: m for m in self.router.catalog.extra})
            self.models = models
        return self.models

    async def call(self, parent, name, args, call_id=None):
        if self.stopping:
            raise ValueError('Adapter is shutting down')
        spec = next((s for s in self.specs if s['name'] == name), None)
        if not spec or not isinstance(args, dict):
            raise ValueError('Unknown bridge agent tool or invalid arguments')
        schema = spec['inputSchema']
        if set(args) - set(schema['properties']) or set(schema['required']) - set(args):
            raise ValueError('Unexpected or missing agent tool arguments')
        for key, value in args.items():
            kind = schema['properties'][key]['type']
            if kind == 'string' and (not isinstance(value, str) or not value.strip() or len(value) > 64000):
                raise ValueError('Agent tool strings must be nonempty and at most 64000 characters')
        if not self.registry.get(parent):
            raise ValueError('Resume the parent task before delegating')
        turn_id = call_id.split(':', 1)[0] if call_id else None
        if turn_id and self.registry.preference('cancelled-turn:' + parent + ':' + turn_id):
            raise ValueError('The parent turn was cancelled; no further child work will be dispatched')
        if name == 'list_models':
            return {'models': list(await self.available_models())}
        if name == 'list_agents':
            return {'agents': [self.public(self.owned(parent, a['id'])) for a in self.registry.agents(parent)]}
        if name == 'wait_agents':
            return await self.wait(parent, args)
        # Serialization and at-most-once dispatch survive concurrent adapters.
        async with self.locks.setdefault(parent, asyncio.Lock()):
            with Lease(self.state, 'agent-call:' + parent):
                key = json.dumps([parent, call_id]) if call_id else None
                previous = self.registry.agent_call(key) if key else None
                if previous:
                    if previous.get('state') != 'done':
                        raise ValueError('An earlier tool dispatch has an uncertain outcome; it will not be repeated')
                    if previous.get('error'):
                        raise ValueError(previous['error'])
                    return previous['result']
                if key:
                    self.registry.agent_call(key, {'state': 'dispatching'})
                try:
                    if name == 'spawn_agent':
                        result = await self.spawn(parent, args)
                    elif name == 'send_message':
                        result = await self.followup(parent, args)
                    else:
                        agent = self.owned(parent, args['target'])
                        await self.interrupt(agent)
                        result = self.public(self.owned(parent, agent['id']))
                except Exception as exc:
                    if key:
                        self.registry.agent_call(key, {'state': 'done', 'error': str(exc)})
                    raise
                if key:
                    self.registry.agent_call(key, {'state': 'done', 'result': result})
                return result

    async def spawn(self, parent, args):
        cancellation = self.registry.preference('agent-cancel:' + parent)
        route = self.registry.get(parent)
        parent_agent = self.registry.agent(parent)
        depth = parent_agent['depth'] + 1 if parent_agent else 1
        root = parent_agent['root_id'] if parent_agent else parent
        model = args.get('model') or route['model']
        entry = (await self.available_models()).get(model)
        if not entry:
            raise ValueError('Unknown model: ' + model + '. Use list_models; no substitute was selected.')
        effort = args.get('reasoning_effort') or route['options'].get('effort')
        supported = [e['reasoningEffort'] for e in entry.get('supportedReasoningEfforts', [])]
        if args.get('reasoning_effort') and effort not in supported:
            raise ValueError('Unsupported reasoning_effort for ' + model + ': ' + ', '.join(supported))
        if effort not in supported:
            effort = entry.get('defaultReasoningEffort')
        with Lease(self.state, 'agent-tree:' + root):
            active = [a for a in self.registry.agents() if a.get('root_id') == root and a['status'] in ACTIVE]
            if depth > 3 or len(active) >= 4:
                raise ValueError('Bridge agent limit: four active children per root task, nesting depth three')
            options = route['options']
            params = {k: copy.deepcopy(options[k]) for k in (
                'approvalPolicy', 'approvalsReviewer', 'permissions', 'runtimeWorkspaceRoots', 'personality'
            ) if options.get(k) is not None}
            if 'permissions' not in params:
                # Full effective policy is also supplied at turn/start, including write roots.
                # Only computed when no named permission profile is being forwarded: the
                # native API rejects a request that combines `permissions` with `sandbox`.
                policy = options.get('sandboxPolicy') or {}
                params['sandbox'] = {'readOnly': 'read-only', 'workspaceWrite': 'workspace-write',
                                     'dangerFullAccess': 'danger-full-access'}.get(policy.get('type'), options.get('sandbox') or 'read-only')
            params.update(cwd=route['cwd'], model=model, ephemeral=False, experimentalRawEvents=False,
                          modelProvider=PROVIDERS[self.router.catalog.engine(model)], allowProviderModelFallback=False,
                          config={'model_reasoning_effort': effort} if effort else {})
            # Inherit only tools the hosting app supplied. Each child receives its
            # own bridge namespaces, and no credentials/config are copied.
            params.update(copy.deepcopy(self.registry.capabilities(parent)))
            response = await self.router.request('thread/start', params)
            tid = response['thread']['id']
            lease = Lease(self.state, 'agent:' + tid)
            if not lease.acquire():
                raise ValueError('New child unexpectedly has an active owner')
            self.leases[tid] = lease
            agent = {'id': tid, 'parent_id': parent, 'root_id': root, 'depth': depth,
                     'task_name': (args.get('task_name') or 'Subagent')[:100], 'model': model,
                     'engine': self.router.catalog.engine(model), 'reasoning_effort': effort,
                     'status': 'starting', 'turn_id': None, 'result': '', 'error': None,
                     'cancellation': cancellation,
                     'updated_at': int(time.time())}
            self.registry.save_agent(agent)
        return await self.dispatch(agent, args['message'], options)

    async def dispatch(self, agent, message, options):
        tid = agent['id']
        params = {k: copy.deepcopy(options[k]) for k in (
            'approvalPolicy', 'approvalsReviewer', 'permissions', 'sandboxPolicy', 'runtimeWorkspaceRoots'
        ) if options.get(k) is not None}
        if 'permissions' in params:
            # Same native constraint as thread/start: a named permission profile cannot
            # be combined with an explicit sandbox policy in the same request.
            params.pop('sandboxPolicy', None)
        params.update(threadId=tid, model=agent['model'], effort=agent['reasoning_effort'],
                      input=[{'type': 'text', 'text': message, 'text_elements': []}])
        try:
            if self.registry.agent(tid).get('cancel_requested') or agent.get('cancellation') != self.registry.preference('agent-cancel:' + agent['parent_id']):
                agent.update(status='interrupted')
                self.registry.save_agent(agent)
                self.leases.pop(tid).close()
                return self.public(agent)
            response = await self.router.request('turn/start', params, delegated=True)
            # Completion may arrive before the RPC response; don't overwrite it.
            fresh = self.registry.agent(tid)
            if fresh['status'] in ACTIVE:
                fresh.update(status='running', turn_id=response['turn']['id'])
                self.registry.save_agent(fresh)
            if self.registry.agent(tid).get('cancel_requested') or agent.get('cancellation') != self.registry.preference('agent-cancel:' + agent['parent_id']):
                await self.interrupt(self.registry.agent(tid))
            return self.public(self.registry.agent(tid))
        except BaseException:
            agent.update(status='uncertain', error={'message': 'Turn dispatch did not acknowledge; no automatic retry.'})
            self.registry.save_agent(agent)
            if tid in self.leases:
                self.leases.pop(tid).close()
            raise

    async def followup(self, parent, args):
        agent = self.owned(parent, args['target'])
        if agent['status'] not in ('completed', 'failed', 'interrupted'):
            raise ValueError('Wait for or stop the child before continuing; uncertain dispatches cannot be replayed')
        lease = Lease(self.state, 'agent:' + agent['id'])
        if not lease.acquire():
            raise ValueError('This child is active in another adapter process')
        try:
            await self.router.request('thread/resume', {'threadId': agent['id'], 'excludeTurns': True})
        except BaseException:
            lease.close()
            raise
        self.leases[agent['id']] = lease
        agent.update(status='starting', turn_id=None, result='', error=None, images=[], cancel_requested=False,
                     cancellation=self.registry.preference('agent-cancel:' + parent))
        self.registry.save_agent(agent)
        return await self.dispatch(agent, args['message'], self.registry.get(agent['id'])['options'])

    async def wait(self, parent, args):
        targets, timeout = args['targets'], args.get('timeout_ms', 10000)
        if not isinstance(targets, list) or not 1 <= len(targets) <= 8 or any(not isinstance(t, str) for t in targets):
            raise ValueError('targets must contain one to eight agent ids')
        if type(timeout) is not int or not 0 <= timeout <= 60000:
            raise ValueError('timeout_ms must be between 0 and 60000')
        end = time.monotonic() + timeout / 1000
        while True:
            agents = [self.owned(parent, tid) for tid in targets]
            active = any(a['status'] in ACTIVE for a in agents)
            if not active or time.monotonic() >= end:
                return {'agents': [self.public(a) for a in agents], 'timed_out': active}
            await asyncio.sleep(min(0.1, max(0, end - time.monotonic())))

    async def interrupt(self, agent):
        if agent['status'] not in ACTIVE:
            await self.cancel_children(agent['id'])
            return
        if agent['id'] not in self.leases:
            raise ValueError('This child is running in another adapter process; stop it in its owning task')
        if agent.get('turn_id') is None:
            agent['cancel_requested'] = True
            self.registry.save_agent(agent)
            return
        await self.router.request('turn/interrupt', {'threadId': agent['id'], 'turnId': agent['turn_id']})

    async def cancel_children(self, parent, turn_id=None):
        if turn_id:
            self.registry.preference('cancelled-turn:' + parent + ':' + turn_id, True)
        self.registry.preference('agent-cancel:' + parent, str(uuid.uuid4()))
        for agent in self.registry.agents(parent):
            if agent['status'] in ACTIVE:
                if agent['status'] == 'starting' and agent.get('turn_id') is None:
                    continue  # Its dispatch observes the cancellation epoch before/after its RPC.
                await self.interrupt(agent)

    async def close(self):
        self.stopping = True
        for tid in list(self.leases):
            with contextlib.suppress(Exception):
                await self.interrupt(self.registry.agent(tid))
        for task in list(self.callbacks):
            task.cancel()
        await asyncio.gather(*self.callbacks, return_exceptions=True)
        for tid, lease in self.leases.items():
            agent = self.registry.agent(tid)
            agent.update(status='uncertain', error={'message': 'Adapter closed before execution outcome was confirmed.'})
            self.registry.save_agent(agent)
            lease.close()
        self.leases.clear()

    async def mcp(self, parent, turn_id, message):
        """Minimal MCP server carried by Claude's official SDK control messages."""
        ident, method, params = message.get('id'), message.get('method'), message.get('params') or {}
        result = {}
        if method == 'initialize':
            result = {'protocolVersion': '2024-11-05', 'capabilities': {'tools': {}},
                      'serverInfo': {'name': NAMESPACE, 'version': '1.0'}}
        elif method == 'tools/list':
            result = {'tools': [{k: v for k, v in s.items() if k != 'type'} for s in self.specs]}
        elif method == 'tools/call':
            try:
                value = await self.call(parent, params.get('name'), params.get('arguments') or {}, f'{turn_id}:mcp:{ident}')
                result = {'content': [{'type': 'text', 'text': json.dumps(value, ensure_ascii=False)}], 'isError': False}
            except Exception as exc:
                result = {'content': [{'type': 'text', 'text': str(exc)}], 'isError': True}
        elif method not in ('ping', 'notifications/initialized', 'notifications/cancelled'):
            return {'jsonrpc': '2.0', 'id': ident, 'error': {'code': -32601, 'message': 'Unsupported bridge MCP method'}}
        return {'jsonrpc': '2.0', 'id': ident, 'result': result}
