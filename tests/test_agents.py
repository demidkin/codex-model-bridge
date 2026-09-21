"""No-network regressions for model selection, ownership and cancellation."""
import asyncio
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock
from bridge.config import Settings
from bridge.leases import Lease
from bridge.registry import Registry
from bridge.router import Router
from test_routes import FakeCore

class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        key = self.root / 'fixture.key'; key.write_text('fixture'); key.chmod(0o600)
        (self.root / 'claude-models.json').write_text(json.dumps([{'value':'sonnet','displayName':'Sonnet','description':'fixture','supportedEffortLevels':['low','high']}]))
        settings = Settings(state=self.root, enable_deepseek=True, enable_claude=True, deepseek_key=key)
        self.registry = Registry(self.root / 'routes.sqlite3')
        self.router = Router(settings, self.registry, AsyncMock())
        self.core = FakeCore(); self.router.core = self.core
        self.router.claude = AsyncMock()
        self.router.claude.request.return_value = {'turn':{'id':'turn-claude'}}
        original = self.core.call
        async def call(method, params):
            if method == 'model/list':
                return {'data':[{'model':'gpt-6-astra','defaultReasoningEffort':'high','supportedReasoningEfforts':[{'reasoningEffort':e} for e in ['low','high','xhigh']]}]}
            if method == 'turn/start':
                self.core.calls.append((method, copy.deepcopy(params)))
                return {'turn':{'id':'turn-' + params['threadId']}}
            if method == 'turn/interrupt':
                self.router.agents.observe({'method':'turn/completed','params':{'threadId':params['threadId'],'turn':{'id':params['turnId'],'status':'interrupted','items':[]}}})
                return {}
            return await original(method, params)
        self.core.call = call
        self.manager = self.router.agents
        self.parent = (await self.router.request('thread/start', {'model':'gpt-6-astra','cwd':str(self.root),'sandbox':'read-only','approvalPolicy':'never'}))['thread']['id']
        await self.router.request('thread/settings/update', {'threadId':self.parent,'effort':'xhigh'})
    async def asyncTearDown(self):
        await self.manager.close()
        self.registry.close(); self.temp.cleanup()
    async def spawn(self, **args):
        return await self.manager.call(self.parent,'spawn_agent',{'message':'bounded fixture task',**args})
    def finish(self, agent, text='done'):
        self.manager.observe({'method':'turn/completed','params':{'threadId':agent['id'],'turn':{'id':agent['turn_id'],'status':'completed','items':[{'type':'agentMessage','text':text}]}}})
    async def test_default_inherits_actual_parent_not_global_preference(self):
        self.registry.preference('menu-model',{'model':'deepseek-flash'})
        a = await self.spawn()
        self.assertEqual((a['model'],a['engine'],a['reasoning_effort']),('gpt-6-astra','openai','xhigh'))
        self.assertEqual(self.registry.get(a['id'])['options']['approvalPolicy'],'never')
        self.assertEqual(self.registry.get(a['id'])['options']['sandbox'],'read-only')
    async def test_each_explicit_engine_and_no_subscription_to_api_conversion(self):
        for model,engine,provider in [('deepseek-flash','deepseek','bridge_deepseek'),('claude-code/sonnet','claude','bridge_claude'),('gpt-6-astra','openai','openai')]:
            a = await self.spawn(model=model)
            self.assertEqual((a['model'],a['engine']), (model,engine))
            self.assertEqual(self.registry.get(a['id'])['provider'],provider)
            self.finish(a)
        native_claude_turns = [p for m,p in self.core.calls if m=='turn/start' and p.get('model','').startswith('claude-code/')]
        self.assertEqual(native_claude_turns,[])
        self.router.claude.request.assert_awaited()
    async def test_deepseek_parent_default_and_claude_parent_explicit_astra(self):
        route = self.registry.get(self.parent)
        for model,engine in [('deepseek-flash','deepseek'),('claude-code/sonnet','claude')]:
            route.update(model=model,engine=engine); self.registry.save(route)
            a = await self.spawn()
            self.assertEqual(a['model'],model); self.finish(a)
        a=await self.spawn(model='gpt-6-astra'); self.assertEqual(a['engine'],'openai')
    async def test_bad_model_effort_and_unauthorized_options_never_dispatch(self):
        count=len(self.core.threads)
        for args in [{'model':'deepseek-nonexistent'},{'model':'gpt-imaginary'},{'model':'deepseek-flash','reasoning_effort':'xhigh'},{'cwd':'/other'},{'approvalPolicy':'never'}]:
            with self.assertRaises(ValueError): await self.spawn(**args)
        self.assertEqual(len(self.core.threads),count)
    async def test_tool_call_id_is_at_most_once(self):
        args={'message':'once','model':'deepseek-flash'}
        a,b=await asyncio.gather(*(self.manager.call(self.parent,'spawn_agent',args,'same-call') for _ in range(2)))
        self.assertEqual(a['id'],b['id']); self.assertEqual(len(self.registry.agents()),1)
    async def test_uncertain_dispatch_never_retries(self):
        original=self.core.call
        async def fail(method,params):
            if method=='turn/start':raise ConnectionError('lost acknowledgement')
            return await original(method,params)
        self.core.call=fail
        for _ in range(2):
            with self.assertRaises((ValueError,ConnectionError)):
                await self.manager.call(self.parent,'spawn_agent',{'message':'once'},'uncertain')
        a=self.registry.agents()[0]
        self.assertEqual(a['status'],'uncertain'); self.assertEqual(len(self.registry.agents()),1)
        with self.assertRaises(ValueError):await self.manager.call(self.parent,'send_message',{'target':a['id'],'message':'retry'})
    async def test_wait_completion_followup_keeps_model_and_result(self):
        a=await self.spawn(model='deepseek-flash')
        pending=await self.manager.call(self.parent,'wait_agents',{'targets':[a['id']],'timeout_ms':0})
        self.assertTrue(pending['timed_out'])
        self.finish(a,'specific result')
        ready=await self.manager.call(self.parent,'wait_agents',{'targets':[a['id']],'timeout_ms':0})
        self.assertEqual(ready['agents'][0]['result'],'specific result')
        b=await self.manager.call(self.parent,'send_message',{'target':a['id'],'message':'continue'})
        self.assertEqual((a['id'],a['model']),(b['id'],b['model']))
    async def test_one_interrupt_preserves_sibling_parent_interrupt_cancels_rest(self):
        a=await self.spawn(); b=await self.spawn(model='deepseek-flash')
        await self.manager.call(self.parent,'interrupt_agent',{'target':a['id']})
        self.assertEqual(self.registry.agent(a['id'])['status'],'interrupted')
        self.assertEqual(self.registry.agent(b['id'])['status'],'running')
        await self.router.request('turn/interrupt',{'threadId':self.parent,'turnId':'parent-turn'})
        self.assertEqual(self.registry.agent(b['id'])['status'],'interrupted')
    async def test_foreign_parent_cannot_wait_or_control_child(self):
        a=await self.spawn()
        with self.assertRaisesRegex(ValueError,'belong'):
            await self.manager.call(a['id'],'interrupt_agent',{'target':a['id']})
    async def test_completion_before_turn_ack_is_not_overwritten(self):
        original=self.core.call
        async def fast(method,params):
            reply=await original(method,params)
            if method=='turn/start':
                self.manager.observe({'method':'turn/completed','params':{'threadId':params['threadId'],'turn':{**reply['turn'],'status':'completed','items':[{'type':'agentMessage','text':'fast'}]}}})
            return reply
        self.core.call=fast
        a=await self.spawn(); self.assertEqual((a['status'],a['result']),('completed','fast'))
    async def test_claude_mcp_uses_same_inheritance_and_ownership(self):
        response=await self.manager.mcp(self.parent,'parent-turn',{'id':1,'method':'tools/call','params':{'name':'spawn_agent','arguments':{'message':'once','model':'deepseek-flash'}}})
        self.assertFalse(response['result']['isError'])
        a=json.loads(response['result']['content'][0]['text']);self.assertEqual(a['model'],'deepseek-flash')
    async def test_recovery_does_not_interrupt_other_process_lease(self):
        turn={'id':'live-claude-turn','status':'inProgress','items':[]}
        self.registry.save_turn('live',turn)
        script="from bridge.leases import Lease; import sys; l=Lease(sys.argv[1],'claude:live'); assert l.acquire(); print('ready',flush=True); sys.stdin.readline()"
        p=subprocess.Popen([sys.executable,'-c',script,str(self.root)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
        try:
            self.assertEqual(p.stdout.readline().strip(),'ready')
            self.registry.recover();self.assertEqual(self.registry.turns('live')[0]['status'],'inProgress')
        finally:
            p.communicate('\n',timeout=5)
        self.registry.recover();self.assertEqual(self.registry.turns('live')[0]['status'],'interrupted')
    async def test_cancel_during_child_creation_prevents_inference(self):
        original=self.core.call;entered=asyncio.Event();release=asyncio.Event()
        async def delayed(method,params):
            result=await original(method,params)
            if method=='thread/start':entered.set();await release.wait()
            return result
        self.core.call=delayed
        task=asyncio.create_task(self.spawn(model='deepseek-flash'))
        await entered.wait()
        await self.manager.cancel_children(self.parent,'parent-turn')
        release.set();agent=await task
        self.assertEqual(agent['status'],'interrupted')
        self.assertFalse(any(m=='turn/start' for m,p in self.core.calls))
    async def test_cancel_during_dispatch_interrupts_after_ack(self):
        original=self.core.call;entered=asyncio.Event();release=asyncio.Event()
        async def delayed(method,params):
            result=await original(method,params)
            if method=='turn/start':entered.set();await release.wait()
            return result
        self.core.call=delayed
        task=asyncio.create_task(self.spawn())
        await entered.wait()
        await self.manager.cancel_children(self.parent,'parent-turn')
        release.set();agent=await task
        self.assertEqual(agent['status'],'interrupted')
    async def test_second_adapter_can_read_but_cannot_claim_active_child(self):
        a=await self.spawn()
        other_registry=Registry(self.root/'routes.sqlite3')
        other_router=Router(self.router.settings,other_registry,AsyncMock());other_router.core=self.core
        try:
            reply=await other_router.agents.call(self.parent,'wait_agents',{'targets':[a['id']],'timeout_ms':0})
            self.assertEqual(reply['agents'][0]['status'],'running')
            with self.assertRaisesRegex(ValueError,'another adapter'):
                await other_router.agents.call(self.parent,'interrupt_agent',{'target':a['id']})
            self.assertEqual(self.registry.agent(a['id'])['status'],'running')
        finally:
            await other_router.agents.close();other_registry.close()
    async def test_inherited_write_roots_and_approval_policy_are_not_dropped(self):
        policy={'type':'workspaceWrite','writableRoots':[str(self.root/'allowed')],'networkAccess':False,'excludeTmpdirEnvVar':True,'excludeSlashTmp':True}
        await self.router.request('thread/settings/update',{'threadId':self.parent,'sandboxPolicy':policy,'approvalPolicy':'untrusted'})
        a=await self.spawn(model='deepseek-flash')
        sent=[p for m,p in self.core.calls if m=='turn/start'][-1]
        self.assertEqual(sent['sandboxPolicy'],policy);self.assertEqual(sent['approvalPolicy'],'untrusted')
    async def test_named_permission_profile_excludes_derived_sandbox(self):
        """A route using a named profile (e.g. danger-full-access) must not also send
        the derived sandbox/sandboxPolicy: the native API rejects combining both."""
        policy={'type':'dangerFullAccess'}
        await self.router.request('thread/settings/update',{'threadId':self.parent,'permissions':':danger-full-access','sandboxPolicy':policy})
        before=len(self.core.calls)
        await self.spawn(model='deepseek-flash')
        started=next(p for m,p in self.core.calls[before:] if m=='thread/start')
        self.assertEqual(started.get('permissions'),':danger-full-access');self.assertNotIn('sandbox',started)
        sent=[p for m,p in self.core.calls if m=='turn/start'][-1]
        self.assertEqual(sent.get('permissions'),':danger-full-access');self.assertNotIn('sandboxPolicy',sent)
    async def test_cancelled_parent_tool_cannot_spawn_late(self):
        await self.manager.cancel_children(self.parent,'cancelled')
        with self.assertRaisesRegex(ValueError,'parent turn was cancelled'):
            await self.manager.call(self.parent,'spawn_agent',{'message':'late'},'cancelled:call-1')
        self.assertEqual(self.registry.agents(),[])
    async def test_child_cannot_be_dispatched_independently_from_second_ui(self):
        a=await self.spawn()
        with self.assertRaisesRegex(ValueError,'managed child task'):
            await self.router.request('turn/start',{'threadId':a['id'],'input':[]})
        self.assertEqual(self.registry.agent(a['id'])['status'],'running')
    async def test_disabled_delegation_replies_to_old_persisted_tool(self):
        self.router.agents=None;self.core.send=AsyncMock()
        await self.router.event({'id':10,'method':'item/tool/call','params':{'namespace':'bridge_agents','threadId':self.parent}})
        self.assertFalse(self.core.send.call_args.args[0]['result']['success'])
