"""Skill discovery, explicit skill input, inherited host tools and image evidence."""
import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock
from bridge.config import Settings, read_models
from bridge.registry import Registry
from bridge.router import Router
from test_routes import FakeCore

class SkillTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        key=self.root/'fake.key';key.write_text('fixture');key.chmod(0o600)
        (self.root/'claude-models.json').write_text(json.dumps([{'value':'sonnet','displayName':'Sonnet','description':'fixture'}]))
        self.skill_root=self.root/'test-skill';self.skill_root.mkdir()
        (self.skill_root/'SKILL.md').write_text('Use this fixture skill. Read references/rule.md.')
        (self.skill_root/'references').mkdir();(self.skill_root/'references/rule.md').write_text('RESOURCE_NONCE_726')
        self.metadata={'name':'fixture-skill','description':'Use when asked to verify fixture resources','path':str(self.skill_root/'SKILL.md'),'enabled':True,'scope':'user'}
        self.registry=Registry(self.root/'state.sqlite3')
        self.settings=Settings(state=self.root,enable_deepseek=True,enable_claude=True,deepseek_key=key)
        self.emit=AsyncMock();self.router=Router(self.settings,self.registry,self.emit)
        self.core=FakeCore();self.router.core=self.core;self.core.send=AsyncMock()
        original=self.core.call
        async def call(method,params):
            if method=='skills/list':
                return {'data':[{'cwd':c,'skills':[copy.deepcopy(self.metadata),{**self.metadata,'name':'disabled','enabled':False}],'errors':[]} for c in params['cwds']]}
            if method=='model/list':return {'data':[{'model':'gpt-6-astra','defaultReasoningEffort':'low','supportedReasoningEfforts':[{'reasoningEffort':'low'}]}]}
            if method=='turn/start':return {'turn':{'id':'turn-'+params['threadId']}}
            return await original(method,params)
        self.core.call=call
        self.router.claude=AsyncMock();self.router.claude.request.return_value={'turn':{'id':'claude-turn'}}
        self.tid=(await self.router.request('thread/start',{'model':'deepseek-flash','cwd':str(self.root),'approvalPolicy':'never','sandbox':'read-only'}))['thread']['id']
    async def asyncTearDown(self):
        await self.router.skills.close()
        for lease in self.router.agents.leases.values():lease.close()
        self.registry.close();self.temp.cleanup()
    async def call(self,name,args=None):return await self.router.skills.call(self.tid,name,args or {},'test-call')
    async def test_all_enabled_skills_are_discovered_without_disabling_user_policy(self):
        skills=(await self.call('list_skills'))['skills'];self.assertEqual([s['name'] for s in skills],['fixture-skill'])
        with self.assertRaisesRegex(ValueError,'unavailable'):await self.call('read_skill',{'name':'disabled'})
    async def test_external_start_and_resume_get_catalogue_and_usage_instructions(self):
        initial=[p for m,p in self.core.calls if m=='thread/start'][0]
        self.assertIn('fixture-skill',initial['developerInstructions']);self.assertIn(str(self.skill_root),initial['developerInstructions'])
        await self.router.request('thread/resume',{'threadId':self.tid})
        resumed=[p for m,p in self.core.calls if m=='thread/resume'][-1]
        self.assertIn('Enabled Codex skill catalogue',resumed['developerInstructions'])
    async def test_read_resources_and_block_traversal_symlink_escape(self):
        text=await self.call('read_resource',{'name':'fixture-skill','path':'references/rule.md'})
        self.assertEqual(text['text'],'RESOURCE_NONCE_726')
        (self.root/'outside.txt').write_text('not a skill resource')
        (self.skill_root/'link').symlink_to(self.root/'outside.txt')
        for path in ['../outside.txt','link',str(self.root/'outside.txt')]:
            with self.assertRaises(ValueError):await self.call('read_resource',{'name':'fixture-skill','path':path})
    async def test_resume_refreshes_catalogue_without_losing_caller_instructions(self):
        tid=(await self.router.request('thread/start',{'model':'deepseek-flash','cwd':str(self.root),'developerInstructions':'Preserve this task rule.'}))['thread']['id']
        self.router.start_options.clear()
        reopened=Registry(self.root/'state.sqlite3')
        self.assertIn('Preserve this task rule.',reopened.task_instructions(tid));reopened.close()
        await self.router.request('thread/resume',{'threadId':tid})
        p=[p for m,p in self.core.calls if m=='thread/resume'][-1]
        self.assertIn('Preserve this task rule.',p['developerInstructions'])
        self.assertEqual(p['developerInstructions'].count('<bridge_codex_skills>'),1)
    async def test_legacy_resume_does_not_overwrite_unknown_native_instructions(self):
        self.registry.db.execute('DELETE FROM task_instructions WHERE thread_id=?',(self.tid,))
        await self.router.request('thread/resume',{'threadId':self.tid})
        p=[p for m,p in self.core.calls if m=='thread/resume'][-1]
        self.assertNotIn('developerInstructions',p)
    async def test_version_three_upgrade_preserves_routes_agents_and_history(self):
        path=self.root/'legacy.sqlite3';old=Registry(path)
        route=self.registry.get(self.tid);old.save(route)
        old.save_agent({'id':'old-child','parent_id':self.tid,'status':'completed','result':'kept'})
        old.db.execute("INSERT INTO claude_turns VALUES ('old-claude','old-turn','{\"status\":\"completed\"}',1)")
        old.db.executescript('DROP TABLE task_capabilities; DROP TABLE task_instructions; PRAGMA user_version=3;')
        old.close()
        upgraded=Registry(path)
        try:
            self.assertEqual(upgraded.get(self.tid)['model'],route['model'])
            self.assertEqual(upgraded.agent('old-child')['result'],'kept')
            self.assertEqual(upgraded.db.execute('SELECT payload FROM claude_turns').fetchone()[0],'{"status":"completed"}')
            self.assertEqual(upgraded.capabilities(self.tid),{})
            self.assertIsNone(upgraded.task_instructions(self.tid))
            self.assertEqual(upgraded.db.execute('PRAGMA user_version').fetchone()[0],4)
        finally:upgraded.close()
    async def test_changes_refresh_and_disable_removed_skill(self):
        await self.call('read_skill',{'name':'fixture-skill'})
        self.metadata['enabled']=False
        with self.assertRaises(ValueError):await self.call('read_skill',{'name':'fixture-skill'})
    async def test_codex_skill_inputs_are_supported_by_claude(self):
        c=(await self.router.request('thread/start',{'model':'claude-code/sonnet','cwd':str(self.root)}))['thread']['id']
        await self.router.request('turn/start',{'threadId':c,'input':[{'type':'text','text':'Use the selected skill'},{'type':'skill','name':'fixture-skill','path':self.metadata['path']}]})
        params=self.router.claude.request.call_args.args[1]
        self.assertTrue(all(i['type']=='text' for i in params['input']))
        self.assertIn('Use this fixture skill',params['input'][1]['text'])
        self.assertIn('fixture-skill',params['_codex_skill_context'])
    async def test_image_input_passes_through_claude_input_unchanged(self):
        c=(await self.router.request('thread/start',{'model':'claude-code/sonnet','cwd':str(self.root)}))['thread']['id']
        image={'type':'localImage','path':'/tmp/fixture.png'}
        await self.router.request('turn/start',{'threadId':c,'input':[{'type':'text','text':'What is in this image?'},image]})
        params=self.router.claude.request.call_args.args[1]
        self.assertEqual(params['input'][1],image)
    async def test_skill_absolute_path_alias_resolves_to_native_catalogue(self):
        alias=self.root/'skill-alias';alias.symlink_to(self.skill_root,target_is_directory=True)
        result=await self.call('read_skill',{'name':str(alias/'SKILL.md')})
        self.assertEqual(result['name'],'fixture-skill')
    async def test_host_dynamic_tools_and_capability_roots_survive_restart_and_inherit(self):
        tool={'type':'function','name':'host_fixture','description':'fixture','inputSchema':{'type':'object'}}
        roots=[{'id':'fixture','location':{'type':'environment','environmentId':'env','path':str(self.skill_root)}}]
        parent=(await self.router.request('thread/start',{'model':'gpt-6-astra','cwd':str(self.root),'dynamicTools':[tool],'selectedCapabilityRoots':roots,'config':{'unrelated':'do not persist'}}))['thread']['id']
        self.assertEqual(self.registry.capabilities(parent),{'dynamicTools':[tool],'selectedCapabilityRoots':roots})
        self.router.start_options.clear()
        child=await self.router.agents.call(parent,'spawn_agent',{'model':'deepseek-flash','message':'fixture'},'child')
        p=[p for m,p in self.core.calls if m=='thread/start'][-1]
        self.assertIn(tool,p['dynamicTools']);self.assertEqual(p['selectedCapabilityRoots'],roots)
        self.assertEqual(sum(t['name']=='bridge_skills' for t in p['dynamicTools']),1)
        self.assertEqual(sum(t['name']=='bridge_agents' for t in p['dynamicTools']),1)
        self.assertEqual(self.registry.capabilities(child['id'])['dynamicTools'],[tool])
    async def test_inherited_host_call_is_forwarded_unchanged(self):
        message={'id':5,'method':'item/tool/call','params':{'namespace':'host_fixture','tool':'x','threadId':self.tid}}
        await self.router.event(message);self.emit.assert_awaited_with(message)
    async def test_imagegen_requires_native_artifact_evidence_not_model_claim(self):
        manager=self.router.agents;manager.call=AsyncMock(side_effect=[{'id':'image-child'},{'timed_out':False}])
        manager.owned=lambda parent,tid:{'status':'completed','result':'I created an image','images':[]}
        with self.assertRaisesRegex(ValueError,'verified saved image'):await self.call('generate_image',{'prompt':'flower'})
        self.assertEqual(manager.call.call_args_list[0].args[2]['model'],'gpt-6-astra')
        self.assertEqual(self.registry.get(self.tid)['model'],'deepseek-flash')
    async def test_imagegen_returns_only_observed_saved_artifact(self):
        image=self.root/'generated.png';image.write_bytes(b'fixture image')
        manager=self.router.agents;manager.call=AsyncMock(side_effect=[{'id':'image-child'},{'timed_out':False}])
        manager.owned=lambda parent,tid:{'status':'completed','images':[{'savedPath':str(image),'status':'completed'}]}
        result=await self.call('generate_image',{'prompt':'flower'})
        self.assertEqual(result['images'],[str(image)])
    async def test_imagegen_cancellation_stops_helper(self):
        manager=self.router.agents
        manager.call=AsyncMock(side_effect=[{'id':'image-child'},asyncio.CancelledError()])
        manager.owned=lambda parent,tid:{'id':tid,'status':'running'}
        manager.interrupt=AsyncMock()
        with self.assertRaises(asyncio.CancelledError):await self.call('generate_image',{'prompt':'flower'})
        manager.interrupt.assert_awaited_once_with({'id':'image-child','status':'running'})
    async def test_failed_image_artifact_is_not_returned(self):
        image=self.root/'failed.png';image.write_bytes(b'partial')
        manager=self.router.agents;manager.call=AsyncMock(side_effect=[{'id':'image-child'},{'timed_out':False}])
        manager.owned=lambda parent,tid:{'status':'completed','images':[{'savedPath':str(image),'status':'failed'}]}
        with self.assertRaisesRegex(ValueError,'verified saved image'):await self.call('generate_image',{'prompt':'flower'})
    async def test_image_events_are_attached_to_selected_child(self):
        child=await self.router.agents.call(self.tid,'spawn_agent',{'message':'fixture'},'event-child')
        self.router.agents.observe({'method':'item/completed','params':{'threadId':child['id'],'item':{'type':'imageGeneration','id':'image','status':'completed','savedPath':str(self.root/'image.png')}}})
        self.assertEqual(self.registry.agent(child['id'])['images'][0]['savedPath'],str(self.root/'image.png'))
    async def test_claude_mcp_reads_same_codex_skill(self):
        result=await self.router.skills.mcp(self.tid,'turn',{'id':1,'method':'tools/call','params':{'name':'read_skill','arguments':{'name':'fixture-skill'}}})
        self.assertFalse(result['result']['isError']);self.assertIn('Use this fixture skill',result['result']['content'][0]['text'])
    def test_deepseek_catalogue_enables_native_skill_instructions(self):
        self.assertTrue(all(m['include_skills_usage_instructions'] for m in read_models(self.settings.deepseek_catalog)))
