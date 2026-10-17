"""Bounded physical control using an immutable reviewed HTTP client snapshot."""
import errno
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

root = Path('/Users/jeremylien/Projects/local-agent-dual-mac')
config = json.loads(sys.stdin.readline(262145))
assert config['role'] == 'reviewed-client-small-policy-control'
assert config['mode'] in ('native-default', 'dual-default', 'dual-laptop-no-residency')
assert config['run_name'] in ('qualified-native-default-v1', 'qualified-dual-default-v1', 'qualified-dual-laptop-policy-v1')
assert config['port'] in (18400, 18402, 18404)
out = root / 'results' / config['run_name']
out.mkdir(exist_ok=False)
server = None
client = None
server_log = None
client_log = None
result = {'status': 'failed', 'role': config['role'], 'mode': config['mode'],
          'started_at': time.time(), 'config': config, 'observations': [], 'client_started': False}

def save(name, value):
    with (out / name).open('x') as stream:
        json.dump(value, stream, indent=2)

def pressure():
    return int(subprocess.run(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'],
                             capture_output=True, text=True, timeout=3, check=True).stdout)

def closed():
    with socket.socket() as sock:
        sock.settimeout(1.5)
        return sock.connect_ex(('127.0.0.1', config['port'])) == errno.ECONNREFUSED

def guard():
    level = pressure()
    result['observations'].append({'timestamp': time.time(), 'pressure': level})
    assert level == 1, 'normal-memory-pressure gate failed'
    if (out / 'server.log').exists():
        assert (out / 'server.log').stat().st_size < 32 * 1024**2, 'server log bound'
    if (out / 'client.log').exists():
        assert (out / 'client.log').stat().st_size < 1024**2, 'client log bound'

def stop(signum, frame):
    raise TimeoutError('bounded control terminated by signal ' + str(signum))

for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGALRM):
    signal.signal(sig, stop)
signal.alarm(235)
try:
    assert closed()
    guard()
    assert subprocess.run(['git', '-C', str(root / 'upstream/llama.cpp'), 'rev-parse', 'HEAD'],
                          capture_output=True, text=True, check=True, timeout=5).stdout.strip() == config['source_revision']
    for name, identity in config['build_files'].items():
        path = root / 'build-api-arm64/bin' / name
        assert path.stat().st_size == identity['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == identity['sha256']
    for item in config['assets']['files']:
        path = Path(config['assets']['model_dir']) / item['name']
        assert path.stat().st_size == item['bytes']
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b''):
                digest.update(block)
        assert digest.hexdigest() == item['sha256']
    snapshot = Path(config['snapshot_dir'])
    snapshot.relative_to(root / 'qualification-snapshots')
    assert snapshot.name == config['snapshot_tree_sha256']
    snapshot_manifest = json.loads((snapshot / 'snapshot-manifest.json').read_text())
    assert snapshot_manifest['files'] == config['snapshot_files']
    for rel, digest in config['snapshot_files'].items():
        assert hashlib.sha256((snapshot / rel).read_bytes()).hexdigest() == digest
    env = dict(os.environ, GGML_RPC_NO_RDMA='1', GGML_RPC_DEBUG='3',
               TMPDIR='/Users/jeremylien/.hermes/cache/scratch', PYTHONDONTWRITEBYTECODE='1')
    env.pop('GGML_METAL_NO_RESIDENCY', None)
    result['mini_policy'] = {'GGML_METAL_NO_RESIDENCY_present': 'GGML_METAL_NO_RESIDENCY' in env,
                             'expected_residency': 'default-enabled; requested environment/source binding, not an independent property log'}
    binary = root / 'build-api-arm64/bin/llama-server'
    d = Path(config['assets']['model_dir'])
    argv = [str(binary)]
    if config['mode'] != 'native-default':
        argv += ['--rpc', config['rpc']]
    argv += ['-m', str(d / 'Qwen3.5-0.8B-Q4_K_M.gguf'), '--mmproj', str(d / 'mmproj-BF16.gguf'),
             '--host', '127.0.0.1', '--port', str(config['port']), '--alias', 'qwen35-vision-control',
             '--ctx-size', '2048', '--parallel', '1', '--batch-size', '256', '--ubatch-size', '256',
             '--gpu-layers', '99', '--device', 'MTL0' if config['mode'] == 'native-default' else 'MTL0,RPC0',
             '--mmproj-device', 'MTL0', '--split-mode', 'none' if config['mode'] == 'native-default' else 'layer']
    if config['mode'] != 'native-default':
        argv += ['--tensor-split', '1,1']
    argv += ['--fit', 'off', '--image-max-tokens', '256', '--no-warmup', '--no-cont-batching',
             '--reasoning', 'off', '--predict', '32', '--seed', '42', '-lv', '4']
    save('argv.json', argv)
    server_log = (out / 'server.log').open('xb')
    server = subprocess.Popen(argv, stdout=server_log, stderr=subprocess.STDOUT,
                              env=env, start_new_session=True)
    result['server_pid'] = server.pid
    save('started.json', result)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 75
    ready = False
    while time.monotonic() < deadline:
        guard()
        assert server.poll() is None, 'server exited before readiness'
        try:
            with opener.open('http://127.0.0.1:' + str(config['port']) + '/health', timeout=1) as response:
                raw = response.read(65537)
                assert len(raw) <= 65536
                health = json.loads(raw)
            if health.get('status') == 'ok':
                ready = True
                break
        except OSError:
            time.sleep(0.2)
    assert ready, 'startup deadline'
    result['ready_at'] = time.time()
    client_argv = [config['client_python'], '-B', '-m', 'qualification.client',
                   '--endpoint', 'http://127.0.0.1:' + str(config['port']),
                   '--manifest', str(snapshot / 'qualification/fixtures/public-cases.json'),
                   '--output', str(out / 'client'), '--model', 'qwen35-vision-control',
                   '--timeout', '45', '--deadline', '150', '--max-response-bytes', '1048576',
                   '--max-tokens', '32']
    save('client-argv.json', client_argv)
    client_log = (out / 'client.log').open('xb')
    client = subprocess.Popen(client_argv, stdout=client_log, stderr=subprocess.STDOUT,
                              cwd=snapshot, env=env, start_new_session=True)
    result['client_started'] = True
    client_deadline = time.monotonic() + 165
    while client.poll() is None:
        guard()
        assert time.monotonic() < client_deadline, 'client supervisor deadline'
        assert server.poll() is None, 'server exited during requests'
        time.sleep(0.25)
    result['client_exit'] = client.returncode
    assert client.returncode == 0, 'client failed; preserve partial HTTP receipt'
    receipt_bytes = (out / 'client/receipt.json').read_bytes()
    receipt = json.loads(receipt_bytes)
    assert receipt['success'] is True and receipt['completed_cases'] == receipt['total_cases'] == 3
    assert receipt['attestation'] is False
    expected = {'red': 'red', 'blue': 'blue', 'split': 'red'}
    assert {c['id'] for c in receipt['cases']} == set(expected)
    answers = []
    for case in receipt['cases']:
        assert case['success'] is True and case['response']['complete'] is True
        assert case['output'].strip() == expected[case['id']], 'incorrect image answer'
        assert case['finish_reason'] == 'stop', 'image answer did not stop normally'
        answers.append({k: case[k] for k in ('id', 'output', 'finish_reason', 'usage', 'request_sha256')})
    props = next(p['observed'] for p in receipt['probes'] if p['path'] == '/props' and p['success'])
    assert props['default_generation_settings']['n_ctx'] == 2048, 'effective context changed'
    result['answers'] = answers
    result['client_receipt_sha256'] = hashlib.sha256(receipt_bytes).hexdigest()
    result['effective_context'] = 2048
    guard()
    result['status'] = 'three-image-control-passed'
except BaseException as exc:
    result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
finally:
    signal.alarm(0)
    for name, process in [('client', client), ('server', server)]:
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=12)
            result[name + '_exit'] = process.returncode
            result[name + '_reaped'] = process.returncode is not None
    for log in (client_log, server_log):
        if log is not None:
            log.close()
    result['listener_closed'] = closed()
    result['post_pressure'] = pressure()
    result['finished_at'] = time.time()
    for name in ('server.log', 'client.log'):
        if (out / name).exists():
            result[name + '_sha256'] = hashlib.sha256((out / name).read_bytes()).hexdigest()
    save('result.json', result)
    print(json.dumps(result, indent=2), flush=True)
sys.exit(0 if result['status'] == 'three-image-control-passed' and result.get('server_reaped')
         and result.get('client_reaped') and result['listener_closed'] and result['post_pressure'] == 1 else 1)
