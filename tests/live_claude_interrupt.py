"""Opt-in interruption of an actually running Claude shell command."""
import asyncio
import json
import tempfile
from pathlib import Path
from bridge.claude import ClaudeEngine
from bridge.config import Settings
from bridge.registry import Registry
from test_routes import FakeCore

async def run():
    with tempfile.TemporaryDirectory(prefix='bridge-claude-interrupt-') as directory:
        root=Path(directory)
        registry=Registry(root/'routes.sqlite3')
        settings=Settings(state=root,enable_claude=True)
        route={'thread_id':'interruption-fixture','engine':'claude','provider':'bridge_claude',
               'model':'claude-code/sonnet','cwd':str(root),'options':{'approvalPolicy':'untrusted','sandbox':'workspace-write'},'locked':True}
        registry.save(route)
        approvals=[]
        async def emit(event):
            if 'id' in event and 'method' in event:
                approvals.append(event['method'])
                engine.accept_response({'id':event['id'],'result':{'decision':'accept'}})
        engine=ClaudeEngine(settings,registry,emit,FakeCore())
        try:
            await engine.start({'effort':'low','input':[{'type':'text','text':
                'Use Bash to run exactly: /usr/bin/touch started.txt; /bin/sleep 8; /usr/bin/touch after-cancel.txt . Do not use other tools.',
                'text_elements':[]}]},route)
            async with asyncio.timeout(120):
                while not (root/'started.txt').exists():
                    if not engine.active:
                        raise RuntimeError('Claude finished before starting the test command')
                    await asyncio.sleep(0.05)
            process=engine.active[route['thread_id']]['process']
            await engine.interrupt(route['thread_id'],None)
            await asyncio.sleep(9)  # Exceed the cancelled command's delayed side effect.
            report={'command_started':True, 'approval_forwarded':bool(approvals),
                    'delayed_side_effect_absent':not (root/'after-cancel.txt').exists(),
                    'official_process_reaped':process.returncode is not None,
                    'turn_status':registry.turns(route['thread_id'])[-1]['status']}
        finally:
            await engine.close()
            registry.close()
        return report
if __name__=='__main__':
    report=asyncio.run(run())
    print(json.dumps(report,indent=2))
    assert report['delayed_side_effect_absent'] and report['official_process_reaped'] and report['turn_status']=='interrupted'
