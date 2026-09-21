"""Explicit live checks: subscription Astra/Claude plus small real DeepSeek requests."""
import argparse
import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from bridge.claude import ClaudeEngine
from bridge.config import Settings
from bridge.registry import Registry
from bridge.router import Router
from bridge.rpc import CoreClient

async def run(claude_parent=False):
    report={'live':True,'gui_verified':False,'scenario':'claude-parent' if claude_parent else 'astra-parent'}
    settings0=Settings.load()
    with tempfile.TemporaryDirectory(prefix='bridge-live-agents-') as temporary:
        root=Path(temporary)
        shutil.copyfile(settings0.state/'claude-models.json',root/'claude-models.json')
        settings=Settings(state=root,enable_deepseek=True,enable_claude=True,deepseek_key=settings0.deepseek_key)
        registry=Registry(root/'routes.sqlite3');completed={};events=[];errors=[]
        async def emit(event):
            if event.get('method')=='turn/completed':completed[event['params']['threadId']]=event['params']['turn']
            if event.get('method')=='error':errors.append(event['params']['error'])
            if event.get('method')=='item/completed':
                item=event['params']['item']
                events.append({'thread_id':event['params']['threadId'],'type':item['type'],'tool':item.get('tool'),'success':item.get('success')})
            if 'id' in event and 'method' in event:
                reply={'id':event['id'],'result':{'decision':'decline'}}
                if not router.claude.accept_response(reply):await core.send(reply)
        router=Router(settings,registry,emit)
        core=CoreClient([settings.core,'app-server','-c','mcp_servers.unityMCP.enabled=false','-c','mcp_servers.node_repl.enabled=false','-c','mcp_servers.computer-use.enabled=false'],router.event)
        router.core=core
        router.claude=ClaudeEngine(settings,registry,router.external_event,core);router.claude.agents=router.agents
        await core.start();tid=None
        try:
            await router.request('initialize',{'clientInfo':{'name':'bridge_live_agents','version':'1'},'capabilities':{'experimentalApi':True}})
            await core.send({'method':'initialized'})
            model='claude-code/sonnet' if claude_parent else 'gpt-6-astra'
            parent=await router.request('thread/start',{'model':model,'cwd':temporary,'approvalPolicy':'never','sandbox':'read-only','ephemeral':False,'experimentalRawEvents':False,'config':{'model_reasoning_effort':'low'}})
            tid=parent['thread']['id']
            if claude_parent:
                prompt='Use the bridge_agents MCP spawn_agent tool to launch exactly one child with model deepseek-flash and message "Reply exactly CLAUDE_CHILD_OK. Do not use any tools or delegate." Then wait_agents for its completed result. Report the returned marker. Use no other tools. This is an authorized bounded integration test.'
            else:
                prompt='This is an authorized bounded integration test. Use only bridge_agents tools. Launch exactly three children with spawn_agent: (1) omit model, message "Reply exactly INHERITED_ASTRA_OK. Do not use tools or delegate."; (2) model deepseek-flash, message "Reply exactly DEEPSEEK_CHILD_OK. Do not use tools or delegate."; (3) model claude-code/sonnet, message "Reply exactly CLAUDE_CHILD_OK. Do not use tools or delegate." Wait for all three using wait_agents, repeating if necessary. Report the three markers and actual models returned by the tool. Do not launch additional children or use native spawn_agent.'
            await router.request('turn/start',{'threadId':tid,'effort':'low','input':[{'type':'text','text':prompt,'text_elements':[]}]})
            async with asyncio.timeout(240):
                while tid not in completed:await asyncio.sleep(0.2)
            agents=registry.agents(tid)
            report['parent']={'model':model,'provider':parent['modelProvider'],'status':completed[tid]['status']}
            report['children']=[{k:a.get(k) for k in ('model','engine','reasoning_effort','status','result','error')} for a in agents]
            report['errors']=errors
            print(json.dumps(report,indent=2),flush=True)
            expected={'deepseek-flash'} if claude_parent else {'gpt-6-astra','deepseek-flash','claude-code/sonnet'}
            assert {a['model'] for a in agents}==expected,report
            assert all(a['status']=='completed' for a in agents),report
            assert completed[tid]['status']=='completed',report
            final=' '.join(i.get('text','') for i in completed[tid].get('items',[]) if i.get('type')=='agentMessage')
            markers=['CLAUDE_CHILD_OK'] if claude_parent else ['INHERITED_ASTRA_OK','DEEPSEEK_CHILD_OK','CLAUDE_CHILD_OK']
            assert all(m in final for m in markers),{'final':final,'report':report}
            report['markers_returned_to_parent']=True
        finally:
            await router.close()
            archived=[]
            for ident in ([tid] if tid else [])+[a['id'] for a in registry.agents()]:
                try:
                    await router.request('thread/archive',{'threadId':ident});archived.append(True)
                except Exception:archived.append(False)
            report['test_tasks_archived']=all(archived)
            await core.close();registry.close()
    return report
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--claude-parent',action='store_true');parser.add_argument('--report')
    args=parser.parse_args();report=asyncio.run(run(args.claude_parent))
    if args.report:Path(args.report).write_text(json.dumps(report,indent=2)+'\n')
