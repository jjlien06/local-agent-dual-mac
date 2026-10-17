import os,json,subprocess,hashlib,sys
from pathlib import Path
os.environ['GGML_METAL_NO_RESIDENCY']='1'
os.environ['GGML_RPC_NO_RDMA']='1'
os.environ['GGML_RPC_DEBUG']='3'
incoming=json.loads(sys.stdin.readline(131073))
assert incoming['role']=='small-no-residency-vision-only'
def policy_pressure():
 return int(subprocess.run(['sysctl','-n','kern.memorystatus_vm_pressure_level'],capture_output=True,text=True,check=True,timeout=3).stdout)
assert policy_pressure()==1
policy_root=Path('/Users/jeremylien/Projects/local-agent-dual-mac')
assert subprocess.run(['git','-C',str(policy_root/'upstream/llama.cpp'),'rev-parse','HEAD'],capture_output=True,text=True,check=True,timeout=5).stdout.strip()==incoming['source_revision']
for filename,identity in incoming['build_files'].items():
 path=policy_root/'build-api-arm64/bin'/filename
 assert path.stat().st_size==identity['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==identity['sha256']
for identity in incoming['assets']['files']:
 path=Path(incoming['assets']['model_dir'])/identity['name'];assert path.stat().st_size==identity['bytes'];h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
 assert h.hexdigest()==identity['sha256']
import subprocess,os
os.environ['GGML_RPC_NO_RDMA']='1'
assert int(subprocess.run(['sysctl','-n','kern.memorystatus_vm_pressure_level'],capture_output=True,text=True,check=True).stdout)==1
list_result=subprocess.run(['/Users/jeremylien/Projects/local-agent-dual-mac/build-api-arm64/bin/llama-server','--rpc','169.254.47.213:18568','--list-devices'],capture_output=True,text=True,timeout=20)
assert list_result.returncode==0 and 'RPC0' in list_result.stdout,list_result.stdout+list_result.stderr
bad_order=subprocess.run(['/Users/jeremylien/Projects/local-agent-dual-mac/build-api-arm64/bin/llama-server','--device','MTL0,RPC0','--rpc','169.254.47.213:18568','--list-devices'],capture_output=True,text=True,timeout=20)
assert bad_order.returncode!=0 and 'invalid device: RPC0' in bad_order.stderr
good_order=subprocess.run(['/Users/jeremylien/Projects/local-agent-dual-mac/build-api-arm64/bin/llama-server','--rpc','169.254.47.213:18568','--device','MTL0,RPC0','--list-devices'],capture_output=True,text=True,timeout=20)
assert good_order.returncode==0,good_order.stdout+good_order.stderr
import json,subprocess,time,socket,errno,urllib.request,hashlib,signal,sys
from pathlib import Path
x=json.loads('{"assets": {"schema": "dual-mac-vision-control-assets-v1", "role": "Small same-architecture VLM control only; NOT the final model recommendation", "repo": "unsloth/Qwen3.5-0.8B-GGUF", "revision": "6ab461498e2023f6e3c1baea90a8f0fe38ab64d0", "files": [{"name": "Qwen3.5-0.8B-Q4_K_M.gguf", "bytes": 532517120, "sha256": "bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517"}, {"name": "mmproj-BF16.gguf", "bytes": 207346528, "sha256": "d312c4d02fd46eea7a16e4f3bbb58840e6222209322ca1e33ca03247ad8935d6"}], "total_bytes": 739863648, "target_model_unchanged": "Qwen/Qwen3.8-27B", "storage_root": "/Volumes/\\u65b0\\u589e\\u78c1\\u789f\\u5340/local-agent-models", "model_dir": "/Volumes/\\u65b0\\u589e\\u78c1\\u789f\\u5340/local-agent-models/qwen35-0.8b-control-6ab461498e20"}, "cases": [{"case_id": "red", "image_sha256": "9b5d2bc0516a2749576352b9c38c6e9f88d2db423d8a187c1c003034b2136b3c", "image_b64": "iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAIAAABMXPacAAABWklEQVR4nO3OQQ0AMBAEofVv+iqDxzRBALvtg/wgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwg7gEgaMOyrMtNTwAAAABJRU5ErkJggg==", "expected": "red"}, {"case_id": "blue", "image_sha256": "32cba78a00615c233d68cbde61d1cb07763e3c7524b1cea9ca0756b336843e86", "image_b64": "iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAIAAABMXPacAAABWklEQVR4nO3OQQ0AMBAEofVv+iqDxzRBANvuo/wgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwgzg/i/CDOD+L8IM4P4vwg7QGY6MOyYa3yZwAAAABJRU5ErkJggg==", "expected": "blue"}], "prompt": "What is the dominant color in this image? Answer with exactly one color word."}')
x['assets']=incoming['assets']
r=Path('/Users/jeremylien/Projects/local-agent-dual-mac');out=r/'results/small-policy-dual-v1';out.mkdir(exist_ok=False)
port=18390;deadline=time.monotonic()+180;p=None
result={'parser_regression':{'bad_order_exit':bad_order.returncode,'bad_order_stderr':bad_order.stderr,'good_order_exit':good_order.returncode},'listed_devices':list_result.stdout,'schema':'small-dual-vision-smoke-v1','status':'failed','role':'Qwen3.5-0.8B architecture control only, not final27B','requests':[],'source_revision':'ea0c4c2c85fb183ec0fbd30a1af4e4275a18a064','model_revision':x['assets']['revision']}
def closed():
 s=socket.socket();s.settimeout(1)
 try:return s.connect_ex(('127.0.0.1',port))==errno.ECONNREFUSED
 finally:s.close()
def stop(signum,frame):raise KeyboardInterrupt('termination signal')
for sig in [signal.SIGTERM,signal.SIGHUP,signal.SIGALRM]:signal.signal(sig,stop)
signal.alarm(210)
try:
 assert closed(),'port already occupied'
 d=Path(x['assets']['model_dir']);binary=r/'build-api-arm64/bin/llama-server'
 argv=[str(binary),'-m',str(d/'Qwen3.5-0.8B-Q4_K_M.gguf'),'--mmproj',str(d/'mmproj-BF16.gguf'),'--host','127.0.0.1','--port',str(port),'--alias','qwen35-vision-control','--ctx-size','2048','--parallel','1','--batch-size','256','--ubatch-size','256','--gpu-layers','99','--device','MTL0,RPC0','--mmproj-device','MTL0','--split-mode','layer','--tensor-split','1,1','--rpc','169.254.47.213:18568','--image-max-tokens','256','--no-warmup','--no-cont-batching','--reasoning','off','--predict','32','--seed','42','-lv','4']
 rpc_i=argv.index('--rpc');rpc_pair=argv[rpc_i:rpc_i+2];del argv[rpc_i:rpc_i+2];argv[1:1]=rpc_pair
 result['argv']=argv;result['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
 with (out/'argv.json').open('x') as f:json.dump(argv,f,indent=2)
 log=(out/'server.log').open('xb');p=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);result['pid']=p.pid
 ready=False
 while time.monotonic()<deadline:
  assert policy_pressure()==1,'pressure guard'
  assert (out/'server.log').stat().st_size<32*1024**2,'log bound'
  if p.poll() is not None:raise RuntimeError('server exited during startup '+str(p.returncode))
  try:
   with urllib.request.urlopen('http://127.0.0.1:'+str(port)+'/health',timeout=1) as resp:
    health=json.loads(resp.read(65536))
   if health.get('status')=='ok':ready=True;break
  except Exception:time.sleep(0.2)
 if not ready:raise TimeoutError('startup deadline')
 for c in x['cases']:
  assert policy_pressure()==1,'request pressure guard'
  body={'model':'qwen35-vision-control','temperature':0,'seed':42,'max_tokens':32,'stream':False,'chat_template_kwargs':{'enable_thinking':False},'messages':[{'role':'user','content':[{'type':'text','text':x['prompt']},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+c['image_b64']}}]}]}
  data=json.dumps(body,separators=(',',':')).encode()
  with (out/(c['case_id']+'-request.json')).open('xb') as f:f.write(data)
  start=time.monotonic();req=urllib.request.Request('http://127.0.0.1:'+str(port)+'/v1/chat/completions',data=data,headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=max(0.1,min(45,deadline-time.monotonic()))) as resp:raw=resp.read(1024**2+1)
  assert len(raw)<=1024**2,'response too large'
  with (out/(c['case_id']+'-response.json')).open('xb') as f:f.write(raw)
  response=json.loads(raw);content=response['choices'][0]['message']['content'];assert isinstance(content,str) and content.strip()
  row={'case_id':c['case_id'],'image_sha256':c['image_sha256'],'request_sha256':hashlib.sha256(data).hexdigest(),'response_sha256':hashlib.sha256(raw).hexdigest(),'content':content,'finish_reason':response['choices'][0].get('finish_reason'),'usage':response.get('usage'),'wall_seconds':time.monotonic()-start,'expected':c['expected'],'color_match':content.strip().lower().strip('.!')==c['expected']}
  result['requests'].append(row)
 result['status']='success' if len(result['requests'])==2 and all(q['color_match'] for q in result['requests']) else 'failed-image-control'
except BaseException as e:result['error']={'type':type(e).__name__,'message':str(e)}
finally:
 if p is not None:
  if p.poll() is None:
   p.terminate()
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:p.kill();p.wait(timeout=10)
  result['exit_code']=p.returncode;result['owned_process_reaped']=p.returncode is not None
  log.close()
 signal.alarm(0)
 result['post_pressure']=policy_pressure()
 result['resource_policy']={'GGML_METAL_NO_RESIDENCY':'1'}
 result['asset_hashes_verified']=incoming['assets']['files']
 result['listener_closed']=closed()
 result['server_log_sha256']=hashlib.sha256((out/'server.log').read_bytes()).hexdigest() if (out/'server.log').exists() else None
 result['result_dir']=str(out)
 with (out/'result.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result,indent=2),flush=True)
sys.exit(0 if result['status']=='success' and result.get('owned_process_reaped') and result['listener_closed'] and result['post_pressure']==1 else 1)
