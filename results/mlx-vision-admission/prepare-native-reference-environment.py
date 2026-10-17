"""Stage a pinned source and isolated resolver environment; no model imports."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path('/Users/jeremylien/Projects/local-agent-dual-mac')
BASE = Path('/Users/jeremylien/Projects/paretoquant-metal/.venv')
SOURCE = ROOT / 'upstream/mlx-vlm'
VENV = ROOT / '.vision-venv'
OUT = ROOT / 'results/native-vision-dependencies-v1'
PIN = 'b8bf04c6dd7f3bd3a2a5033bbfecef6375c3bfb0'
receipt = {'schema': 'native-vision-reference-dependency-probe-v1',
           'started_at': time.time(), 'status': 'failed', 'operations': [],
           'packages_installed': False, 'model_imported': False, 'gpu_execution': False}
OUT.mkdir(exist_ok=False)
protected = [BASE / 'lib/python3.13/site-packages/mlx_lm/models/qwen3_5.py',
             BASE / 'lib/python3.13/site-packages/mlx_lm/models/qwen3_next.py']
before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
env = dict(os.environ, TMPDIR='/Users/jeremylien/.hermes/cache/scratch',
           PYTHONDONTWRITEBYTECODE='1', PIP_DISABLE_PIP_VERSION_CHECK='1')
commands = []
try:
    available = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    if available < 12 * 1024**3:
        raise RuntimeError('12GiB disk reserve required before resolver staging')
    if SOURCE.exists() or VENV.exists():
        raise RuntimeError('new source/environment target already exists; no overwrite')
    SOURCE.mkdir(parents=True)
    commands = [
        ('source-init', ['git', 'init', str(SOURCE)], 20),
        ('source-remote', ['git', '-C', str(SOURCE), 'remote', 'add', 'origin',
                           'https://github.com/Blaizzy/mlx-vlm.git'], 20),
        ('source-fetch', ['git', '-C', str(SOURCE), 'fetch', '--depth=1', 'origin', PIN], 90),
        ('source-checkout', ['git', '-C', str(SOURCE), 'checkout', '--detach', PIN], 30),
        ('isolated-venv', [str(BASE / 'bin/python'), '-B', '-m', 'venv', str(VENV)], 50),
        ('environment-prefix', [str(VENV / 'bin/python'), '-B', '-c',
                                'import sys;print(sys.prefix)'], 10),
        ('resolve-only', [str(VENV / 'bin/python'), '-B', '-m', 'pip', 'install',
                          '--dry-run', '--no-cache-dir', '--report', str(OUT / 'pip-resolve.json'),
                          'mlx==0.32.4', 'mlx-lm==0.32.0', 'transformers==5.18.0',
                          'numpy==2.5.3', str(SOURCE)], 180),
    ]
    for label, argv, deadline in commands:
        log = OUT / (label + '.log')
        start = time.time()
        with log.open('xb') as f:
            result = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT,
                                    env=env, timeout=deadline)
        receipt['operations'].append({'label': label, 'exit_code': result.returncode,
                                      'seconds': time.time() - start, 'log': str(log),
                                      'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest()})
        if result.returncode != 0:
            receipt['failed_step'] = label
            raise RuntimeError(label + ' failed; see retained log')
        if label == 'source-checkout':
            actual = subprocess.run(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'],
                                    capture_output=True, text=True, check=True, timeout=10).stdout.strip()
            if actual != PIN:
                raise ValueError('source pin mismatch')
            language = SOURCE / 'mlx_vlm/models/qwen3_5/language.py'
            h = hashlib.sha256(language.read_bytes()).hexdigest()
            if h != '6cfeb5e9efee8042e9d82bbc696168d0e15cb95220365613caa84ea2568304a7':
                raise ValueError('reviewed language source mismatch')
            receipt['source_revision'] = actual
            receipt['language_source_sha256'] = h
            receipt['requirements_text'] = (SOURCE / 'requirements.txt').read_text()
        if label == 'environment-prefix' and log.read_text().strip() != str(VENV):
            raise RuntimeError('environment is not isolated target')
    report = json.loads((OUT / 'pip-resolve.json').read_text())
    receipt['resolved_packages'] = [{'name': p['metadata']['name'],
                                     'version': p['metadata']['version']}
                                    for p in report['install']]
    receipt['status'] = 'source-staged-and-isolated-dependencies-resolved-not-installed'
except Exception as exc:
    receipt['error'] = {'type': type(exc).__name__, 'message': str(exc)}
finally:
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    receipt['protected_language_source_hashes_before'] = before
    receipt['protected_language_source_hashes_after'] = after
    receipt['protected_language_sources_unchanged'] = after == before
    receipt['finished_at'] = time.time()
    with (OUT / 'result.json').open('x') as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps({k: v for k, v in receipt.items() if k not in
                      ('requirements_text', 'protected_language_source_hashes_before',
                       'protected_language_source_hashes_after', 'resolved_packages')}, indent=2))
    print('RESOLVED_PACKAGE_COUNT', len(receipt.get('resolved_packages', [])))
sys.exit(0 if receipt['status'].endswith('resolved-not-installed') else 1)
