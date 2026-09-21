"""Opt-in official Claude subscription transport and permission checks."""
import asyncio
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
    report = {'engine': 'official Claude Code', 'auth': 'claude.ai subscription'}
    with tempfile.TemporaryDirectory(prefix='bridge-claude-live-') as directory:
        root = Path(directory)
        shutil.copyfile(ROOT / '.runtime/claude-models.json', root / 'claude-models.json')
        settings = Settings(state=root, enable_claude=True)
        registry = Registry(root / 'routes.sqlite3')
        completed = asyncio.Event()
        seen, events = [], []
        decision = 'decline'
        async def emit(event):
            events.append(event)
            if event.get('method') == 'turn/completed':
                completed.set()
            if 'id' in event and 'method' in event:
                seen.append(event['method'])
                reply = {'id': event['id'], 'result': {'decision': decision}}
                if not router.claude.accept_response(reply):
                    await core.send(reply)
        router = Router(settings, registry, emit)
        core = CoreClient([settings.core, 'app-server', '-c', 'mcp_servers.unityMCP.enabled=false',
            '-c', 'mcp_servers.node_repl.enabled=false', '-c', 'mcp_servers.computer-use.enabled=false'], router.event)
        router.core = core
        router.claude = ClaudeEngine(settings, registry, emit, core)
        tid = None
        await core.start()
        async def turn(prompt):
            completed.clear()
            await router.request('turn/start', {'threadId': tid, 'model': 'claude-code/sonnet', 'effort': 'low',
                'input': [{'type': 'text', 'text': prompt, 'text_elements': []}]})
            await asyncio.wait_for(completed.wait(), 150)
            last = registry.turns(tid)[-1]
            text = '\n'.join(i.get('text','') for i in last['items'] if i['type'] == 'agentMessage')
            return last, text
        try:
            await router.request('initialize', {'clientInfo': {'name': 'bridge_claude_check', 'version': '0.1'}, 'capabilities': {'experimentalApi': True}})
            await core.send({'method': 'initialized'})
            result = await router.request('thread/start', {'model': 'claude-code/sonnet', 'cwd': str(root),
                'ephemeral': False, 'approvalPolicy': 'untrusted', 'sandbox': 'workspace-write', 'experimentalRawEvents': False})
            tid = result['thread']['id']
            first, text = await turn('Remember the nonce CORAL_741. Reply exactly CLAUDE_BRIDGE_OK. Do not use tools.')
            report['first_status'] = first['status']
            report['acknowledgement'] = 'CLAUDE_BRIDGE_OK' in text
            report['first_error'] = first.get('error')
            if first['status'] != 'completed':
                return report
            listed = await router.request('thread/list', {'limit': 100, 'cwd': str(root), 'modelProviders': None})
            report['native_sidebar_visible'] = any(t['id'] == tid for t in listed['data'])
            # The next turn starts a new native process with --resume of the exact session.
            second, text = await turn('What nonce did I ask you to remember? Reply with the nonce only, no tools.')
            report['resume_status'] = second['status']
            report['context_preserved'] = 'CORAL_741' in text
            denied, text = await turn("Use Bash to run exactly: /usr/bin/touch denied-proof.txt . If permission is denied, stop and say DENIED_OK; do not try another tool or command.")
            report['denial_status'] = denied['status']
            report['approval_forwarded'] = 'item/commandExecution/requestApproval' in seen
            report['denied_file_absent'] = not (root / 'denied-proof.txt').exists()
            decision = 'accept'
            accepted, text = await turn("Use Bash to run exactly: /usr/bin/printf 'CLAUDE_TOOL_OK\\n' . Then say DONE. No other tools.")
            report['accepted_status'] = accepted['status']
            report['command_output_received'] = any('CLAUDE_TOOL_OK' in i.get('aggregatedOutput', '') for i in accepted['items'])
            history = await router.request('thread/read', {'threadId': tid, 'includeTurns': True})
            report['restored_turn_count'] = len(history['thread']['turns'])
            report['streaming_received'] = any(e.get('method') == 'item/agentMessage/delta' for e in events)
        finally:
            await router.close()
            if tid:
                try:
                    await core.call('thread/archive', {'threadId': tid})
                    report['test_task_archived'] = True
                except Exception:
                    report['test_task_archived'] = False
            await core.close()
            registry.close()
    return report
if __name__ == '__main__':
    report = asyncio.run(run())
    print(json.dumps(report, indent=2))
    assert all(report.get(key) for key in ('acknowledgement', 'context_preserved', 'approval_forwarded',
        'denied_file_absent', 'command_output_received', 'streaming_received', 'test_task_archived'))
    assert all(report.get(key) == 'completed' for key in ('first_status','resume_status','denial_status','accepted_status'))
