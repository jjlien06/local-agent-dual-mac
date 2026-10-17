import json,subprocess,shlex,hashlib
from pathlib import Path
r=Path('/Users/jeremylien/Projects/local-agent-dual-mac');o=r/'results/qualified-native-default-v1';o.mkdir(exist_ok=False)
p=json.loads(Path('/Users/jeremylien/Projects/paretoquant-metal/results/qwen38-speed-v2/fast-profile-v2.json').read_text())['parameters']
b=json.loads((r/'results/full-target-admission/mini-build-snapshot.json').read_text());source=(r/'results/full-target-admission/qualified-policy-remote-v1.py').read_text();deploy=json.loads((r/'results/qualification-mini-deployment-controller.json').read_text());assert deploy['exit']==0;snap=json.loads(deploy['stdout']);assert snap['test_exit']==0
x={'role':'reviewed-client-small-policy-control','mode':'native-default','run_name':'qualified-native-default-v1','port':18400,'rpc':None,'source_revision':b['source_revision'],'build_files':b['files'],'assets':json.loads((r/'qualification-assets.json').read_text()),'remote_source_sha256':hashlib.sha256(source.encode()).hexdigest(),'snapshot_dir':snap['snapshot_dir'],'snapshot_tree_sha256':snap['tree_sha256'],'snapshot_files':snap['hashes_read_back'],'client_python':snap['interpreter']}
with (o/'admitted-config.json').open('x') as f:json.dump(x,f,indent=2)
q=subprocess.run(['ssh','-T','-i',p['ssh_key'],'-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=8','-o','HostKeyAlias='+p['ssh_host_key_alias'],p['ssh_target'],'TMPDIR=/Users/jeremylien/.hermes/cache/scratch '+shlex.join(['/Users/jeremylien/Projects/local-agent-dual-mac/.build-venv/bin/python','-u','-c',source])],input=json.dumps(x)+chr(10),capture_output=True,text=True,timeout=280)
with (o/'remote-controller.log').open('xb') as f:f.write(q.stdout.encode()+q.stderr.encode())
print(q.stdout);print(q.stderr);raise SystemExit(q.returncode)
