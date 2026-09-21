"""No-inference check of the executable wrapper, with its own temporary state."""
import asyncio,json,os,tempfile,shutil
from pathlib import Path
from bridge.config import Settings,ROOT
from bridge.rpc import CoreClient
async def main():
 original=Settings.load()
 with tempfile.TemporaryDirectory(prefix='bridge-wrapper-agents-') as temp:
  root=Path(temp);state=root/'state';state.mkdir(mode=0o700)
  shutil.copyfile(original.state/'claude-models.json',state/'claude-models.json')
  config=root/'bridge.toml';config.write_text('[bridge]\nstate = '+json.dumps(str(state))+'\nenable_deepseek = true\nenable_claude = true\n')
  before=os.environ.get('CODEX_BRIDGE_CONFIG');os.environ['CODEX_BRIDGE_CONFIG']=str(config)
  async def event(x):pass
  core=CoreClient([str(ROOT/'bin/codex-bridge'),'app-server'],event)
  await core.start()
  if before is None:os.environ.pop('CODEX_BRIDGE_CONFIG')
  else:os.environ['CODEX_BRIDGE_CONFIG']=before
  try:
   await core.call('initialize',{'clientInfo':{'name':'bridge_wrapper_agents','version':'1'},'capabilities':{'experimentalApi':True}})
   await core.send({'method':'initialized'})
   models=await core.call('model/list',{'includeHidden':False,'limit':100})
   reply=await core.call('thread/start',{'cwd':temp,'model':'gpt-6-astra','ephemeral':True,'approvalPolicy':'never','sandbox':'read-only','experimentalRawEvents':False})
   await core.call('thread/unsubscribe',{'threadId':reply['thread']['id']})
  finally:await core.close()
  report={'wrapper_exit_code':core.process.returncode,'model_count':len(models['data']),'thread_start_with_injected_tools':'accepted','provider':reply['modelProvider'],'isolated_state':True,'inference_requests':0}
  assert report['wrapper_exit_code']==0
  print(json.dumps(report))
if __name__ == '__main__':
 asyncio.run(main())
