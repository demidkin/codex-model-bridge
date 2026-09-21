"""Real bundled core + deterministic local Responses server: delegation end to end."""
import asyncio
import json
import tempfile
import threading
from pathlib import Path
from http.server import ThreadingHTTPServer
from native_smoke import ResponsesFixture
from bridge.config import Settings
from bridge.registry import Registry
from bridge.router import Router
from bridge.rpc import CoreClient

class AgentFixture(ResponsesFixture):
    requests=[]
    counts={}
    def do_POST(self):
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        inputs=body.get('input',[])
        prompt=' '.join(c.get('text','') for i in inputs if i.get('role')=='user' and isinstance(i.get('content'),list) for c in i['content'])
        parent='BRIDGE_PARENT_PROBE' in prompt
        self.requests.append({'model':body['model'],'parent':parent,
            'delegation_tools':any(t.get('name')=='bridge_agents' for t in body.get('tools',[])),
            'skill_tools':any(t.get('name')=='bridge_skills' for t in body.get('tools',[])),
            'host_tools':any(t.get('name')=='host_fixture' for t in body.get('tools',[])),
            'project_skill_context':'native-fixture-skill' in json.dumps(body.get('input',[]))+str(body.get('instructions',''))})
        key=(body['model'],prompt)
        index=self.counts.get(key,0);self.counts[key]=index+1
        if parent and index==0:
            args={'message':'BRIDGE_CHILD_PROBE'}
            if 'EXPLICIT' in prompt:args['model']='deepseek-v4-pro'
            item={'type':'function_call','id':'fc_spawn','call_id':'call_spawn','name':'spawn_agent',
                  'namespace':'bridge_agents','arguments':json.dumps(args),'status':'completed'}
        elif parent and index==1:
            outputs=[i for i in inputs if i.get('type')=='function_call_output']
            output=outputs[-1]['output']
            if isinstance(output,list):output=''.join(x.get('text','') for x in output)
            try:agent=json.loads(output)
            except Exception:raise ValueError('Unexpected dynamic tool reply: '+str(output))
            item={'type':'function_call','id':'fc_wait','call_id':'call_wait','name':'wait_agents','namespace':'bridge_agents',
                  'arguments':json.dumps({'targets':[agent['id']],'timeout_ms':10000}),'status':'completed'}
        elif not parent and index==0:
            item={'type':'function_call','id':'fc_host','call_id':'call_host','name':'read_nonce',
                  'namespace':'host_fixture','arguments':'{}','status':'completed'}
        else:
            text='PARENT_GOT_RESULT' if parent else 'CHILD_RESULT_OK'
            if parent:
                output=[i for i in inputs if i.get('type')=='function_call_output'][-1]['output']
                assert 'CHILD_RESULT_OK' in str(output),str(output)
            else:
                output=[i for i in inputs if i.get('type')=='function_call_output'][-1]['output']
                assert 'HOST_RESOURCE_5297' in str(output),str(output)
            item={'type':'message','id':'msg_'+str(len(self.requests)),'role':'assistant','status':'completed',
                  'content':[{'type':'output_text','text':text,'annotations':[]}]}
        response={'id':'resp_'+str(len(self.requests)),'object':'response','status':'completed','model':body['model'],
                  'output':[item],'usage':{'input_tokens':1,'output_tokens':1,'total_tokens':2,
                                         'input_tokens_details':{'cached_tokens':0},'output_tokens_details':{'reasoning_tokens':0}}}
        events=[{'type':'response.created','response':{**response,'status':'in_progress','output':[]}},
                {'type':'response.output_item.added','output_index':0,'item':{**item,'status':'in_progress'}},
                {'type':'response.output_item.done','output_index':0,'item':item},
                {'type':'response.completed','response':response}]
        payload=''.join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
        self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Content-Length',str(len(payload)))
        self.end_headers();self.wfile.write(payload)

async def run():
    server=ThreadingHTTPServer(('127.0.0.1',0),AgentFixture)
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    class FixtureSettings(Settings):
        def provider_config(self,engine):
            config=super().provider_config(engine)
            if engine=='deepseek':config['model_providers.bridge_deepseek']['base_url']=f'http://127.0.0.1:{server.server_port}'
            return config
    report={'transport':'real native core; local API fixture only'}
    with tempfile.TemporaryDirectory(prefix='bridge-agents-native-') as temp:
        root=Path(temp);key=root/'fixture.key';key.write_text('bridge-fixture-key');key.chmod(0o600)
        skill=root/'.agents/skills/native-fixture-skill';skill.mkdir(parents=True)
        (skill/'SKILL.md').write_text('---\nname: native-fixture-skill\ndescription: Use for native fixture checks.\n---\nRead fixture tools when requested.\n')
        settings=FixtureSettings(state=root,enable_deepseek=True,deepseek_key=key)
        registry=Registry(root/'routes.sqlite3');completed={};errors=[];host_tools=[]
        async def emit(event):
            if event.get('method')=='turn/completed':completed[event['params']['threadId']]=event['params']['turn']
            if event.get('method')=='error':errors.append(event['params'])
            if event.get('method')=='item/tool/call':
                host_tools.append(event)
                assert event['params']['namespace']=='host_fixture',event
                await core.send({'id':event['id'],'result':{'success':True,'contentItems':[{'type':'inputText','text':'HOST_RESOURCE_5297'}]}})
                return
            if 'id' in event and 'method' in event:
                await core.send({'id':event['id'],'result':{'decision':'decline'}})
        async def start():
            router=Router(settings,registry,emit)
            core=CoreClient([settings.core,'app-server','-c','mcp_servers.unityMCP.enabled=false','-c','mcp_servers.node_repl.enabled=false','-c','mcp_servers.computer-use.enabled=false'],router.event)
            router.core=core;await core.start()
            await router.request('initialize',{'clientInfo':{'name':'bridge_agent_probe','version':'1'},'capabilities':{'experimentalApi':True}})
            await core.send({'method':'initialized'})
            return router,core
        router,core=await start();parents=[]
        try:
            for explicit in (False,True):
                host={'type':'namespace','name':'host_fixture','description':'Native fixture host tools','tools':[
                    {'type':'function','name':'read_nonce','description':'Read fixture nonce','inputSchema':{'type':'object','properties':{},'additionalProperties':False}}]}
                p=await router.request('thread/start',{'cwd':temp,'model':'deepseek-flash','approvalPolicy':'never','sandbox':'read-only','ephemeral':False,'experimentalRawEvents':False,'dynamicTools':[host]})
                tid=p['thread']['id'];parents.append(tid)
                if explicit:
                    # Persist a newly tool-equipped task and resume on another adapter.
                    try:await core.call('thread/read',{'threadId':tid,'includeTurns':True})
                    except Exception as exc:
                        if str(exc)!='list_turns is not supported yet':raise
                    await router.close();await core.close();router,core=await start()
                    await router.request('thread/resume',{'threadId':tid,'excludeTurns':True})
                await router.request('turn/start',{'threadId':tid,'input':[{'type':'text','text':'BRIDGE_PARENT_PROBE '+('EXPLICIT' if explicit else 'DEFAULT'),'text_elements':[]}]})
                async with asyncio.timeout(60):
                    while tid not in completed:await asyncio.sleep(0.05)
                assert completed[tid]['status']=='completed',errors
                a=registry.agents(tid)
                assert len(a)==1 and a[0]['status']=='completed',a
                expected='deepseek-v4-pro' if explicit else 'deepseek-flash'
                assert a[0]['model']==expected,a
                assert 'CHILD_RESULT_OK' in a[0]['result'],a
                report['explicit_after_restart' if explicit else 'inheritance']={'model':a[0]['model'],'child_completed':True,'parent_completed':True,'result_returned':True}
            assert len(host_tools)==2,host_tools
            assert all(registry.agent(e['params']['threadId']) for e in host_tools)
            assert all(r['host_tools'] and r['skill_tools'] and r['project_skill_context'] for r in AgentFixture.requests),AgentFixture.requests
            report['requests']=AgentFixture.requests
            report['own_tools_not_forwarded_to_host']=True
            report['inherited_host_tools_called_by_children']=len(host_tools)
            report['skills_visible_in_parent_and_child_inference']=True
        finally:
            for tid in parents+[a['id'] for a in registry.agents()]:
                try:await core.call('thread/archive',{'threadId':tid})
                except Exception:pass
            await router.close();await core.close();registry.close()
            server.shutdown();server.server_close();worker.join()
    return report
if __name__=='__main__':print(json.dumps(asyncio.run(run()),indent=2))
