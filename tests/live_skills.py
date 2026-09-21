"""Opt-in real skill use by DeepSeek/Claude and optional subscription ImageGen service."""
import argparse
import asyncio
import json
import shutil
import tempfile
import uuid
from pathlib import Path
from bridge.config import Settings
from bridge.registry import Registry
from bridge.router import Router
from bridge.rpc import CoreClient
from bridge.claude import ClaudeEngine

async def run(with_image, skip_deepseek=False, report_path=None):
    original=Settings.load();report={'live':True,'gui_verified':False,'image_requested':with_image,'scenarios':[]}
    with tempfile.TemporaryDirectory(prefix='bridge-live-skills-') as temp:
        root=Path(temp);state=root/'state';state.mkdir(mode=0o700)
        shutil.copyfile(original.state/'claude-models.json',state/'claude-models.json')
        skill=root/'.agents/skills/orbital-garden';skill.mkdir(parents=True)
        (skill/'references').mkdir()
        nonce='GARDEN_'+uuid.uuid4().hex[:12]
        (skill/'SKILL.md').write_text('---\nname: orbital-garden\ndescription: Use for orbital garden calibration requests and their verification code.\n---\nFor an orbital garden calibration, read references/verification.txt in this skill directory and return its exact contents. Do not invent the code.\n')
        (skill/'references/verification.txt').write_text(nonce)
        settings=Settings(state=state,enable_deepseek=True,enable_claude=True,deepseek_key=original.deepseek_key)
        registry=Registry(state/'routes.sqlite3');completed={};items={};parents=[]
        async def emit(event):
            p=event.get('params') or {}
            if event.get('method')=='turn/completed':completed[p['threadId']]=p['turn']
            if event.get('method')=='item/completed':
                item=p['item'];items.setdefault(p['threadId'],[]).append({k:item[k] for k in ('type','tool','namespace','status','savedPath') if k in item})
            if 'id' in event and 'method' in event:
                reply={'id':event['id'],'result':{'decision':'decline'}}
                if not router.claude.accept_response(reply):await core.send(reply)
        router=Router(settings,registry,emit)
        core=CoreClient([settings.core,'app-server','-c','mcp_servers.unityMCP.enabled=false','-c','mcp_servers.node_repl.enabled=false','-c','mcp_servers.computer-use.enabled=false'],router.event)
        router.core=core;router.claude=ClaudeEngine(settings,registry,router.external_event,core)
        router.claude.agents=router.agents;router.claude.skills=router.skills
        await core.start()
        async def start(model):
            r=await router.request('thread/start',{'cwd':temp,'model':model,'ephemeral':False,'approvalPolicy':'never','sandbox':'read-only','experimentalRawEvents':False,'config':{'model_reasoning_effort':'low'}})
            tid=r['thread']['id'];parents.append(tid);return tid
        async def turn(tid,inputs):
            completed.pop(tid,None)
            await router.request('turn/start',{'threadId':tid,'effort':'low','input':inputs})
            async with asyncio.timeout(180):
                while tid not in completed:await asyncio.sleep(0.1)
            t=completed[tid];assert t['status']=='completed',t.get('error')
            return '\n'.join(i.get('text','') for i in t.get('items',[]) if i.get('type')=='agentMessage')
        try:
            await router.request('initialize',{'clientInfo':{'name':'bridge_live_skills','version':'1'},'capabilities':{'experimentalApi':True}});await core.send({'method':'initialized'})
            catalogue=await router.skills.catalogue(temp,refresh=True)
            assert any(s['name']=='orbital-garden' for s in catalogue['skills']),[s['name'] for s in catalogue['skills']]
            prompt='Perform orbital garden calibration and return its verification code. Use the appropriate available Codex skill. Do not modify files or delegate. The code must come from the skill resource, not a guess.'
            parent=await start('deepseek-flash')
            if not skip_deepseek:
                text=await turn(parent,[{'type':'text','text':prompt,'text_elements':[]}]);assert nonce in text,text
                report['scenarios'].append({'kind':'deepseek-main-auto-skill','model':'deepseek-flash','resource_nonce_verified':True,'tools':items.get(parent,[])})
                print('DeepSeek main: skill resource verified',flush=True)
                child=await router.agents.call(parent,'spawn_agent',{'model':'deepseek-flash','message':prompt},'skill-child')
                while True:
                    r=await router.agents.call(parent,'wait_agents',{'targets':[child['id']],'timeout_ms':10000})
                    if not r['timed_out']:break
                assert nonce in r['agents'][0]['result'],r
                report['scenarios'].append({'kind':'deepseek-child-auto-skill','model':child['model'],'resource_nonce_verified':True,'tools':items.get(child['id'],[])})
                print('DeepSeek child: skill resource verified',flush=True)
            claude=await start('claude-code/sonnet')
            text=await turn(claude,[{'type':'text','text':'Use the selected Codex skill and return its verification code. Do not modify files or delegate.','text_elements':[]},
                                   {'type':'skill','name':'orbital-garden','path':str(skill/'SKILL.md')}]);assert nonce in text,text
            report['scenarios'].append({'kind':'claude-main-explicit-codex-skill','model':'claude-code/sonnet','resource_nonce_verified':True,'tools':items.get(claude,[])})
            print('Claude: native Codex skill input and resource verified',flush=True)
            if with_image:
                child=await router.agents.call(parent,'spawn_agent',{'model':'deepseek-flash','reasoning_effort':'low','task_name':'ImageGen skill verification',
                    'message':'Use the imagegen skill and generate one small square illustration: a tiny blue robot watering a yellow flower on the Moon, warm sunlight, no text. Use the bridge_skills.generate_image service for built-in ImageGen access. Do not use CLI/API fallback or install anything. Return the real generated image path.'},'image-child')
                async with asyncio.timeout(700):
                    while True:
                        r=await router.agents.call(parent,'wait_agents',{'targets':[child['id']],'timeout_ms':10000})
                        if not r['timed_out']:break
                assert r['agents'][0]['status']=='completed',r
                helpers=registry.agents(child['id'])
                images=[i for a in helpers for i in a.get('images',[]) if i.get('savedPath') and Path(i['savedPath']).is_file()]
                assert images,{'child':r,'helpers':helpers}
                report['scenarios'].append({'kind':'deepseek-child-imagegen-via-astra','model':child['model'],'helper_models':[h['model'] for h in helpers],
                                           'native_images':images,'child_result':r['agents'][0]['result']})
                print('DeepSeek ImageGen: actual image verified',flush=True)
        finally:
            await router.close()
            archived=[]
            for tid in parents+[a['id'] for a in registry.agents()]:
                try:
                    await router.request('thread/archive',{'threadId':tid})
                    archived.append({'id':tid,'archived':True})
                except Exception as exc:archived.append({'id':tid,'archived':False,'error':str(exc)})
            report['test_tasks_archived']=all(x['archived'] for x in archived)
            report['archive_details']=archived
            await core.close();registry.close()
            if report_path:Path(report_path).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--image',action='store_true');parser.add_argument('--skip-deepseek',action='store_true');parser.add_argument('--report',required=True)
    args=parser.parse_args();report=asyncio.run(run(args.image,args.skip_deepseek,args.report))
    print(json.dumps(report,ensure_ascii=False,indent=2))
