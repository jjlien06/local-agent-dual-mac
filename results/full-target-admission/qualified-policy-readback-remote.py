import json,hashlib,socket,errno,subprocess,time
from pathlib import Path
root=Path('/Users/jeremylien/Projects/local-agent-dual-mac/results');runs={};total=0
for run in ['qualified-native-default-v1','qualified-dual-default-v1','qualified-dual-laptop-policy-v1']:
 d=root/run;names=['result.json','argv.json','client-argv.json','started.json','server.log','client.log'];names += ['client/'+p.name for p in sorted((d/'client').iterdir())];assert len(names)<=48
 files={}
 for name in names:
  p=d/name;assert p.is_file() and not p.is_symlink();data=p.read_bytes();total+=len(data);assert len(data)<=2*1024**2 and total<=8*1024**2;files[name]={'text':data.decode('utf-8'),'sha256':hashlib.sha256(data).hexdigest()}
 runs[run]=files
ports={}
for port in [18400,18402,18404]:
 with socket.socket() as s:s.settimeout(1.5);ports[str(port)]=s.connect_ex(('127.0.0.1',port))
assert all(v==errno.ECONNREFUSED for v in ports.values())
p=subprocess.run(['pgrep','-x','llama-server'],capture_output=True,text=True,timeout=5);assert p.returncode==1
h=subprocess.run(['sysctl','kern.memorystatus_vm_pressure_level','vm.swapusage'],capture_output=True,text=True,check=True,timeout=5).stdout
print(json.dumps({'runs':runs,'timestamp':time.time(),'total_artifact_bytes':total,'loopback_ports_connect_errno':ports,'workers_absent':True,'health':h}))