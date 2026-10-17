"""Fixed 64 MiB allocation/free diagnostic; no weights or inference."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
os.environ['GGML_RPC_NO_RDMA']='1'
os.environ['GGML_RPC_DEBUG']='3'
r=Path('/Users/jeremylien/Projects/local-agent-dual-mac')
x=json.loads(sys.stdin.readline(131073))
assert x['role']=='rpc-allocation-accounting-only'
assert x['port']==18384
out=r/'results/rpc-allocation-64m-v1';out.mkdir(exist_ok=False)
result={'role':x['role'],'status':'failed','weights_loaded':False,'inference_executed':False,'events':[],'config':x}
buf=None;base=None

def pressure():
 return int(subprocess.run(['sysctl','-n','kern.memorystatus_vm_pressure_level'],capture_output=True,text=True,check=True,timeout=3).stdout)
def event(name,**fields):
 result['events'].append({'name':name,'timestamp':time.time(),**fields})
 with (out/(str(len(result['events']))+'-'+name+'.json')).open('x') as f:json.dump(result['events'][-1],f)
def stop(signum,frame):raise TimeoutError('allocation diagnostic deadline/signal')
for sig in [signal.SIGALRM,signal.SIGTERM,signal.SIGHUP]:signal.signal(sig,stop)
signal.alarm(45)
try:
 assert pressure()==1
 rev=subprocess.run(['git','-C',str(r/'upstream/llama.cpp'),'rev-parse','HEAD'],capture_output=True,text=True,check=True,timeout=5).stdout.strip();assert rev==x['source_revision']
 for filename,identity in x['build_files'].items():
  p=r/'build-api-arm64/bin'/filename;assert p.stat().st_size==identity['bytes'];assert hashlib.sha256(p.read_bytes()).hexdigest()==identity['sha256']
 base=ctypes.CDLL(str(r/'build-api-arm64/bin/libggml-base.0.27.0.dylib'),mode=ctypes.RTLD_GLOBAL)
 rpc=ctypes.CDLL(str(r/'build-api-arm64/bin/libggml-rpc.0.27.0.dylib'),mode=ctypes.RTLD_GLOBAL)
 rpc.ggml_backend_rpc_buffer_type.argtypes=[ctypes.c_char_p,ctypes.c_uint32];rpc.ggml_backend_rpc_buffer_type.restype=ctypes.c_void_p
 base.ggml_backend_buft_alloc_buffer.argtypes=[ctypes.c_void_p,ctypes.c_size_t];base.ggml_backend_buft_alloc_buffer.restype=ctypes.c_void_p
 base.ggml_backend_buffer_get_size.argtypes=[ctypes.c_void_p];base.ggml_backend_buffer_get_size.restype=ctypes.c_size_t
 base.ggml_backend_buffer_free.argtypes=[ctypes.c_void_p];base.ggml_backend_buffer_free.restype=None
 buft=rpc.ggml_backend_rpc_buffer_type(x['rpc'].encode('ascii'),0);assert buft
 event('before_alloc',requested_bytes=67108864)
 start=time.monotonic();buf=base.ggml_backend_buft_alloc_buffer(buft,67108864);assert buf
 size=base.ggml_backend_buffer_get_size(buf);assert size>=67108864 and size<=67108864+4096
 event('allocated',actual_buffer_bytes=size,wall_seconds=time.monotonic()-start)
 assert pressure()==1
 time.sleep(6)
 base.ggml_backend_buffer_free(buf);buf=None
 event('freed')
 time.sleep(2)
 result['status']='fixed-allocation-and-free-completed'
except BaseException as e:result['error']={'type':type(e).__name__,'message':str(e)}
finally:
 signal.alarm(0)
 if buf and base is not None:base.ggml_backend_buffer_free(buf)
 result['owned_buffer_freed']=buf is None
 result['final_pressure']=pressure()
 with (out/'result.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result,indent=2),flush=True)
sys.exit(0 if result['status']=='fixed-allocation-and-free-completed' and result['owned_buffer_freed'] and result['final_pressure']==1 else 1)
