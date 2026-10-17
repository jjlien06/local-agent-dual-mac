import json,hashlib,subprocess,shlex
from pathlib import Path
r=Path('/Users/jeremylien/Projects/local-agent-dual-mac');p=json.loads(Path('/Users/jeremylien/Projects/paretoquant-metal/results/qwen38-speed-v2/fast-profile-v2.json').read_text())['parameters'];code=(r/'results/full-target-admission/qualified-policy-readback-remote.py').read_text()
q=subprocess.run(['ssh','-T','-i',p['ssh_key'],'-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=8','-o','HostKeyAlias='+p['ssh_host_key_alias'],p['ssh_target'],shlex.join(['/Users/jeremylien/Projects/local-agent-dual-mac/.build-venv/bin/python','-c',code])],capture_output=True,text=True,timeout=45);assert q.returncode==0,q.stdout[:2000]+q.stderr
b=json.loads(q.stdout);summary={k:v for k,v in b.items() if k!='runs'};summary['hashes']={}
for run,files in b['runs'].items():
 d=r/'results'/run/'ssh-readback';d.mkdir(exist_ok=False)
 for name,e in files.items():
  data=e['text'].encode();assert hashlib.sha256(data).hexdigest()==e['sha256'];path=d/name;path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as f:f.write(data)
 rr=json.loads(files['result.json']['text']);assert rr['status']=='three-image-control-passed' and rr['server_reaped'] and rr['client_reaped'] and rr['listener_closed'] and rr['post_pressure']==1
 assert rr['server.log_sha256']==files['server.log']['sha256'];assert rr['client.log_sha256']==files['client.log']['sha256'];assert rr['client_receipt_sha256']==files['client/receipt.json']['sha256']
 summary['hashes'][run]={k:v['sha256'] for k,v in files.items()}
with (r/'results/qualified-dual-laptop-policy-v1/remote-readback.json').open('x') as f:json.dump(summary,f,indent=2)
print(json.dumps({k:v for k,v in summary.items() if k!='hashes'},indent=2));print('HASH_VERIFIED_COMPLETE_RUNS',len(summary['hashes']))
