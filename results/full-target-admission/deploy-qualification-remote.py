import sys,json,base64,hashlib,subprocess,time
from pathlib import Path
raw=sys.stdin.buffer.read(1024**2+1);assert len(raw)<=1024**2
x=json.loads(raw);hashes={k:v['sha256'] for k,v in x['files'].items()};assert hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest()==x['tree_sha256']
r=Path('/Users/jeremylien/Projects/local-agent-dual-mac');d=r/'qualification-snapshots'/x['tree_sha256'];d.mkdir(parents=True,exist_ok=False)
for name,e in x['files'].items():
 rel=Path(name);assert not rel.is_absolute() and '..' not in rel.parts
 b=base64.b64decode(e['base64'],validate=True);assert len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256'];p=d/rel;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(b)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==e['sha256']
with (d/'snapshot-manifest.json').open('x') as f:json.dump({'tree_sha256':x['tree_sha256'],'files':hashes,'scope':'Exact reviewed client/test/fixture snapshot; deployment readback performed'},f,indent=2)
py='/Users/jeremylien/Projects/paretoquant-dual-host/.venv/bin/python'
p=subprocess.run([py,'-B','-m','pytest','tests/test_qualification.py','-q','-p','no:cacheprovider','--basetemp','/Users/jeremylien/.hermes/cache/scratch/qualification-mini-transport-'+x['tree_sha256'][:12]],cwd=d,capture_output=True,text=True,timeout=150)
v=subprocess.run([py,'-c','import sys;print(sys.version)'],capture_output=True,text=True,check=True,timeout=5).stdout
receipt={'snapshot_dir':str(d),'tree_sha256':x['tree_sha256'],'hashes_read_back':hashes,'interpreter':py,'python_version':v,'test_exit':p.returncode,'test_stdout':p.stdout,'test_stderr':p.stderr,'timestamp':time.time(),'no_gpu_or_model_execution':True}
with (r/'results/qualification-mini-snapshot-verification.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt,indent=2));sys.exit(p.returncode)