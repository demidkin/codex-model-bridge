"""Real native metadata persistence with fixture history; no model inference."""
import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from bridge.claude import ClaudeEngine
from bridge.config import ROOT, Settings
from bridge.registry import Registry
from bridge.router import Router
from bridge.rpc import CoreClient, RpcError

async def run():
    with tempfile.TemporaryDirectory(prefix='bridge-claude-metadata-') as directory:
        root = Path(directory)
        shutil.copyfile(ROOT/'.runtime/claude-models.json', root/'claude-models.json')
        settings = Settings(state=root, enable_claude=True)
        registry = Registry(root/'routes.sqlite3')
        async def emit(event):
            pass
        router = Router(settings, registry, emit)
        core = CoreClient([settings.core,'app-server'],router.event)
        router.core = core
        router.claude = ClaudeEngine(settings,registry,emit,core)
        await core.start()
        tid = None
        report = {'model_inference':False}
        try:
            await router.request('initialize',{'clientInfo':{'name':'bridge_metadata','version':'0.1'},'capabilities':{'experimentalApi':True}})
            await core.send({'method':'initialized'})
            initial=await router.request('thread/start',{'cwd':str(root),'model':'claude-code/sonnet','experimentalRawEvents':False})
            tid=initial['thread']['id']
            try:
                await core.call('thread/read',{'threadId':tid,'includeTurns':True})
            except RpcError as exc:
                if str(exc)!='list_turns is not supported yet': raise
            registry.save_turn(tid,{'id':'fixture','items':[],'status':'completed','startedAt':1789910000,'completedAt':1789910001})
            await core.call('thread/unsubscribe',{'threadId':tid})
            await router.request('thread/resume',{'threadId':tid})
            report['cwd_resolves_identically']=Path(registry.get(tid)['cwd']).resolve()==root.resolve()
            for archived in [False,True,False]:
                if archived:
                    await router.request('thread/archive',{'threadId':tid})
                elif report.get('archived_visible'):
                    await router.request('thread/unarchive',{'threadId':tid})
                data=await router.request('thread/list',{'cwd':[str(root)],'archived':archived,'limit':100})
                report['archived_visible' if archived else 'active_visible']=any(t['id']==tid for t in data['data'])
            report['stored_archived_count']=len(registry.claude_threads(True))
        finally:
            if tid:
                try: await core.call('thread/archive',{'threadId':tid})
                except Exception: pass
            await core.close()
            registry.close()
        return report
if __name__=='__main__':
    report = asyncio.run(run())
    print(json.dumps(report,indent=2))
    assert report['cwd_resolves_identically'] and report['active_visible'] and report['archived_visible']
    assert report['stored_archived_count'] == 0
