import hashlib
import json
from pathlib import Path
import shlex
import subprocess

r = Path('/Users/jeremylien/Projects/local-agent-dual-mac')
p = json.loads(Path('/Users/jeremylien/Projects/paretoquant-metal/results/qwen38-speed-v2/fast-profile-v2.json').read_text())['parameters']
source = (r / 'results/mlx-vision-admission/verify-retained-vision-remote.py').read_text()
admission = (r / 'results/mlx-vision-admission/readback-admission.json').read_text()
command = 'TMPDIR=/Users/jeremylien/.hermes/cache/scratch ' + shlex.join([
    '/Users/jeremylien/Projects/paretoquant-dual-host/.venv/bin/python', '-B', '-u', '-c', source])
q = subprocess.run(['ssh', '-T', '-i', p['ssh_key'], '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes',
                    '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=8',
                    '-o', 'HostKeyAlias=' + p['ssh_host_key_alias'], p['ssh_target'], command],
                   input=admission, capture_output=True, text=True, timeout=220)
with (r / 'results/mlx-vision-admission/verification-controller-v1.json').open('x') as stream:
    json.dump({'exit_code': q.returncode, 'stdout': q.stdout, 'stderr': q.stderr,
               'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
               'admission_sha256': hashlib.sha256(admission.encode()).hexdigest()}, stream, indent=2)
print(q.stdout)
print(q.stderr)
raise SystemExit(q.returncode)
