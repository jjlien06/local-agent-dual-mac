"""Bounded install of frozen dependencies in a new, isolated VLM environment."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

WORK = Path('/Users/jeremylien/Projects/local-agent-dual-mac')
OUT = Path(__file__).resolve().parent
ENV = WORK / '.vlm-reference-venv'
PYTHON = Path('/Users/jeremylien/Projects/paretoquant-metal/.venv/bin/python')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    admission = json.loads((OUT/'admission.json').read_text())
    assert sha(OUT/'requirements-hashed.txt') == admission['requirements_sha256']
    assert not ENV.exists(), 'Never reuse or replace an existing environment silently'
    old = json.loads((WORK/'results/native-vision-dependencies-v1/result.json').read_text())
    protected = old['protected_language_source_hashes_after']
    for name, expected in protected.items():
        assert sha(Path(name)) == expected
    env = dict(os.environ, TMPDIR='/Users/jeremylien/.hermes/cache/scratch',
               PYTHONDONTWRITEBYTECODE='1', PIP_DISABLE_PIP_VERSION_CHECK='1')
    started = time.monotonic()
    stages = []
    commands = [
        ('create-isolated-env', [str(PYTHON), '-B', '-m', 'venv', str(ENV)], 40),
        ('install-locked-wheels', [str(ENV/'bin/python'), '-B', '-m', 'pip', 'install',
          '--no-deps', '--require-hashes', '--only-binary=:all:', '--timeout', '15',
          '--retries', '1', '--disable-pip-version-check', '-r', str(OUT/'requirements-hashed.txt')], 180),
        ('pip-check', [str(ENV/'bin/python'), '-B', '-m', 'pip', 'check'], 20),
        ('package-metadata', [str(ENV/'bin/python'), '-B', '-c',
          'import importlib.metadata as m,json;print(json.dumps({d.metadata["Name"].lower().replace("_","-"):d.version for d in m.distributions()}))'], 20),
    ]
    status = 'failed'
    try:
        for label, command, timeout in commands:
            log = OUT/(label+'.log')
            tick = time.monotonic()
            with log.open('xb') as stream:
                completed = subprocess.run(command, env=env, cwd=WORK, stdout=stream,
                                           stderr=subprocess.STDOUT, timeout=timeout)
            stages.append(dict(label=label, exit_code=completed.returncode,
                               seconds=time.monotonic()-tick, log_sha256=sha(log)))
            assert completed.returncode == 0, label+' failed; see preserved log'
        metadata = json.loads((OUT/'package-metadata.log').read_text())
        for package in admission['packages']:
            assert metadata[package['name'].lower().replace('_','-')] == package['version']
        assert all(sha(Path(name)) == expected for name, expected in protected.items())
        status = 'installed-dependencies-verified-source-import-not-yet-exercised'
    finally:
        result = dict(schema='isolated-native-vlm-environment-install-v1', status=status,
                      passed=status.startswith('installed'), stages=stages,
                      elapsed_seconds=time.monotonic()-started,
                      environment=str(ENV), dependency_wheels=admission['wheel_count'],
                      packages_installed=True if status.startswith('installed') else 'partial_or_unknown',
                      protected_language_sources_unchanged=all(sha(Path(name)) == expected for name, expected in protected.items()),
                      existing_numeric_environment_modified=False,
                      model_imported=False, gpu_execution=False,
                      scope='Dependency installation and metadata verification only; pinned source must still be imported and tested. Not VLM inference or capacity admission.')
        with (OUT/'install-result.json').open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
