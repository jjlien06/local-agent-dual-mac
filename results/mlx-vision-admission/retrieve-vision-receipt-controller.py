import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

r = Path('/Users/jeremylien/Projects/local-agent-dual-mac')
previous = json.loads((r / 'results/mlx-vision-admission/verification-controller-v1.json').read_text())
if previous['exit_code'] != 0:
    raise ValueError('verification controller did not complete')
expected = json.loads(previous['stdout'])['result_sha256']
p = json.loads(Path('/Users/jeremylien/Projects/paretoquant-metal/results/qwen38-speed-v2/fast-profile-v2.json').read_text())['parameters']
code = "import hashlib,json,subprocess;from pathlib import Path;p=Path('/Users/jeremylien/Projects/local-agent-dual-mac/results/mlx-vision-readback-v1/result.json');data=p.read_bytes();assert len(data)<1024**2;health=subprocess.run(['sysctl','kern.memorystatus_vm_pressure_level','vm.swapusage'],capture_output=True,text=True,check=True,timeout=5).stdout;print(json.dumps({'text':data.decode(),'sha256':hashlib.sha256(data).hexdigest(),'post_health':health}))"
q = subprocess.run(['ssh', '-T', '-i', p['ssh_key'], '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes',
                    '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=8',
                    '-o', 'HostKeyAlias=' + p['ssh_host_key_alias'], p['ssh_target'],
                    shlex.join(['/Users/jeremylien/Projects/paretoquant-dual-host/.venv/bin/python', '-B', '-c', code])],
                   capture_output=True, text=True, timeout=35)
if q.returncode != 0 or len(q.stdout.encode()) > 2 * 1024**2:
    raise ValueError('bounded receipt retrieval failed')
b = json.loads(q.stdout)
raw = b['text'].encode()
if hashlib.sha256(raw).hexdigest() != b['sha256'] or b['sha256'] != expected:
    raise ValueError('retrieved receipt hash mismatch')
d = json.loads(raw)
if (d['status'] != 'whole-shards-and-vision-payloads-verified' or len(d['vision']) != 333
        or d['full_tensor_count'] != 2180 or d['text_tensor_count'] != 1847
        or sum(x['bytes'] for x in d['vision'].values()) != 921460192
        or len(d['shards']) != 3 or not all(x['pressure'] == 1 for x in d['observations'])):
    raise ValueError('receipt inventory or pressure mismatch')
for item in d['vision'].values():
    if item['dtype'] != 'BF16' or len(item['sha256']) != 64:
        raise ValueError('vision dtype/digest mismatch')
with (r / 'results/mlx-vision-admission/retained-weight-readback.json').open('xb') as f:
    f.write(raw)
summary = {'timestamp': time.time(), 'result_sha256': b['sha256'], 'result_bytes': len(raw),
           'shards_fully_rehashed': 3, 'full_tensor_count': 2180, 'language_tensor_count': 1847,
           'vision_tensor_count': 333, 'vision_BF16_bytes': 921460192,
           'all_sampled_pressure_normal': True, 'post_health': b['post_health'],
           'gpu_execution': False, 'model_imported': False,
           'previous_readback_attempt': 'Large JSON passed through tool-output channel failed parsing; no receipt was published by that attempt. This controller parses and saves full SSH stdout before emitting a small summary.',
           'scope': 'Current read-only checkpoint file integrity; not GPU visual compatibility or model/task qualification.'}
with (r / 'results/mlx-vision-admission/retained-weight-readback-binding.json').open('x') as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
