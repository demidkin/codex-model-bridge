"""Opt-in native checks for Claude sidebar persistence, files and cancellation."""
import asyncio
import copy
import json
import shutil
import tempfile
from pathlib import Path
from bridge.claude import ClaudeEngine
from bridge.config import ROOT, Settings
from bridge.registry import Registry
from bridge.router import Router
from bridge.rpc import CoreClient

async def run():
    report = {}
    with tempfile.TemporaryDirectory(prefix='bridge-claude-lifecycle-') as directory:
        root = Path(directory)
        shutil.copyfile(ROOT / '.runtime/claude-models.json', root / 'claude-models.json')
        settings = Settings(state=root, enable_claude=True)
        registry = Registry(root / 'routes.sqlite3')
        completed, asked = asyncio.Event(), asyncio.Event()
        events, approval_ids = [], []
        decision = 'accept'
        tid = None
        async def emit(event):
            events.append(copy.deepcopy(event))
            if event.get('method') == 'turn/completed':
                completed.set()
            if 'id' in event and 'method' in event:
                approval_ids.append(event['id'])
                asked.set()
                if decision == 'wait':
                    return
                reply = {'id': event['id'], 'result': {'decision': decision}}
                if not router.claude.accept_response(reply):
                    await core.send(reply)
        async def start_core():
            router = Router(settings, registry, emit)
            core = CoreClient([settings.core, 'app-server', '-c', 'mcp_servers.unityMCP.enabled=false',
                '-c', 'mcp_servers.node_repl.enabled=false', '-c', 'mcp_servers.computer-use.enabled=false'], router.event)
            router.core = core
            router.claude = ClaudeEngine(settings, registry, emit, core)
            await core.start()
            await router.request('initialize', {'clientInfo': {'name':'bridge_claude_lifecycle','version':'0.1'},'capabilities':{'experimentalApi':True}})
            await core.send({'method':'initialized'})
            return router, core
        router, core = await start_core()
        try:
            result = await router.request('thread/start', {'model':'claude-code/sonnet','cwd':str(root), 'ephemeral':False,
                'approvalPolicy':'untrusted','sandbox':'workspace-write','experimentalRawEvents':False})
            tid = result['thread']['id']
            await router.request('turn/start', {'threadId':tid,'model':'claude-code/sonnet','effort':'low', 'input':[
                {'type':'text','text':'Use Write to create proof.txt containing exactly FILE_OK. Do not use Bash. Then reply DONE.','text_elements':[]}]})
            await asyncio.wait_for(completed.wait(), 120)
            report['file_written'] = (root / 'proof.txt').read_text() == 'FILE_OK' if (root/'proof.txt').exists() else False
            report['file_approval'] = any(e.get('method') == 'item/fileChange/requestApproval' for e in events)
            report['file_diff_received'] = any(e.get('method') == 'item/completed' and e['params']['item']['type'] == 'fileChange' and
                'FILE_OK' in e['params']['item']['changes'][0]['diff'] for e in events)
            await router.close()
            await core.close()
            registry.close()
            registry = Registry(root / 'routes.sqlite3')
            registry.recover()
            router, core = await start_core()
            listing = await router.request('thread/list', {'limit':100,'cwd':str(root)})
            report['sidebar_after_restart'] = any(t['id'] == tid for t in listing['data'])
            resume = await router.request('thread/resume', {'threadId':tid,'model':'gpt-6-astra'})
            report['provider_after_restart'] = resume['modelProvider']
            report['history_after_restart'] = len(resume['thread']['turns'])
            decision = 'wait'
            asked.clear()
            completed.clear()
            result = await router.request('turn/start', {'threadId':tid,'model':'claude-code/sonnet','effort':'low', 'input':[
                {'type':'text','text':'Use Bash to run /usr/bin/touch cancelled.txt . Do not use other tools.','text_elements':[]}]})
            await asyncio.wait_for(asked.wait(), 120)
            state = router.claude.active[tid]
            process = state['process']
            await router.request('turn/interrupt', {'threadId':tid,'turnId':result['turn']['id']})
            report['cancel_status'] = registry.turns(tid)[-1]['status']
            report['cancelled_file_absent'] = not (root/'cancelled.txt').exists()
            report['process_reaped'] = process.returncode is not None
            report['no_pending_approvals'] = not router.claude.approvals
            await router.request('thread/archive', {'threadId':tid})
            listing = await router.request('thread/list', {'limit':100,'cwd':str(root)})
            report['archive_hides_task'] = not any(t['id']==tid for t in listing['data'])
            listing = await router.request('thread/list', {'limit':100,'cwd':str(root),'archived':True})
            report['archive_preserves_task'] = any(t['id']==tid for t in listing['data'])
        finally:
            await router.close()
            if tid:
                try:
                    await core.call('thread/archive', {'threadId':tid})
                except Exception:
                    pass
            await core.close()
            registry.close()
    return report
if __name__ == '__main__':
    report = asyncio.run(run())
    print(json.dumps(report, indent=2))
    assert all(report[key] for key in ('file_written','file_approval','file_diff_received','sidebar_after_restart',
        'cancelled_file_absent','process_reaped','no_pending_approvals','archive_hides_task','archive_preserves_task'))
    assert report['cancel_status'] == 'interrupted' and report['provider_after_restart'] == 'bridge_claude'
    assert report['history_after_restart'] == 1
