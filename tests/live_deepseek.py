"""Explicit real API smoke test; never prints the key or provider response body."""
import asyncio
import json
import tempfile
from pathlib import Path
from bridge.config import Settings
from bridge.registry import Registry
from bridge.router import Router
from bridge.rpc import CoreClient

async def run():
    with tempfile.TemporaryDirectory(prefix='bridge-deepseek-live-') as directory:
        root = Path(directory)
        settings = Settings(state=root, enable_deepseek=True)
        settings.check_deepseek_key()
        registry = Registry(root / 'routes.sqlite3')
        completed = asyncio.Event()
        report = {'provider': 'bridge_deepseek', 'model': 'deepseek-flash', 'real_api': True}
        async def emit(event):
            method = event.get('method')
            if method == 'item/completed':
                item = event['params']['item']
                if item['type'] == 'agentMessage':
                    report['acknowledgement'] = 'DEEPSEEK_BRIDGE_OK' in item.get('text', '')
            if method == 'turn/completed':
                report['status'] = event['params']['turn']['status']
                error = event['params']['turn'].get('error') or {}
                report['error_kind'] = error.get('codexErrorInfo')
                completed.set()
            if 'id' in event and 'method' in event:
                await core.send({'id': event['id'], 'result': {'decision': 'decline'}})
        router = Router(settings, registry, emit)
        core = CoreClient([settings.core, 'app-server', '-c', 'mcp_servers.unityMCP.enabled=false',
            '-c', 'mcp_servers.node_repl.enabled=false', '-c', 'mcp_servers.computer-use.enabled=false'], router.event)
        router.core = core
        await core.start()
        try:
            await router.request('initialize', {'clientInfo': {'name': 'bridge_live_check', 'version': '0.1'}, 'capabilities': {'experimentalApi': True}})
            await core.send({'method': 'initialized'})
            result = await router.request('thread/start', {'model': 'deepseek-flash', 'cwd': str(root), 'ephemeral': True,
                'approvalPolicy': 'never', 'sandbox': 'read-only', 'experimentalRawEvents': False})
            await router.request('turn/start', {'threadId': result['thread']['id'], 'model': 'deepseek-flash', 'effort': 'low',
                'input': [{'type': 'text', 'text': 'Reply exactly DEEPSEEK_BRIDGE_OK. Do not use tools.', 'text_elements': []}]})
            await asyncio.wait_for(completed.wait(), 100)
        finally:
            await core.close()
            registry.close()
        return report
if __name__ == '__main__':
    report = asyncio.run(run())
    print(json.dumps(report, indent=2))
    assert report.get('acknowledgement') and report.get('status') == 'completed'
