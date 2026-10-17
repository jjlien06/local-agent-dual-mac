"""One-shot full-target loading diagnostic; no model requests or persistence."""
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

os.environ['GGML_RPC_NO_RDMA'] = '1'
root = Path('/Users/jeremylien/Projects/local-agent-dual-mac')
config = json.loads(sys.stdin.readline(131073))
assert config['role'] == 'full-target-load-only'
assert config['port'] == 18382
out = root / 'results/full-target-load-v1'
out.mkdir(exist_ok=False)
port = config['port']
process = None
log = None
result = {'status': 'failed', 'role': config['role'], 'inference_executed': False,
          'observations': [], 'started_at': time.time(), 'config': config}

def save(name, value):
    with (out / name).open('x') as stream:
        json.dump(value, stream, indent=2)

def closed():
    with socket.socket() as sock:
        sock.settimeout(1.5)
        return sock.connect_ex(('127.0.0.1', port)) == errno.ECONNREFUSED

def pressure():
    return int(subprocess.run(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'],
                             capture_output=True, text=True, timeout=3, check=True).stdout)

def stop(signum, frame):
    raise TimeoutError('bounded diagnostic terminated by signal ' + str(signum))

for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGALRM):
    signal.signal(sig, stop)
signal.alarm(330)
try:
    assert closed(), 'API port already occupied'
    assert pressure() == 1, 'preload pressure not normal'
    revision = subprocess.run(['git', '-C', str(root / 'upstream/llama.cpp'), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    assert revision == config['source_revision']
    provision_bytes = (root / 'results/full-target-assets-provision.json').read_bytes()
    provision = json.loads(provision_bytes)
    assert provision['status'] == 'downloaded-and-hash-verified'
    assert provision['revision'] == config['assets']['revision']
    assert {Path(x['file']).name: (x['bytes'], x['sha256']) for x in provision['files']} == {
        x['name']: (x['bytes'], x['sha256']) for x in config['assets']['files']}
    for item in config['assets']['files']:
        path = Path(config['assets']['model_dir']) / item['name']
        assert path.stat().st_size == item['bytes']
    result['provision_receipt_sha256'] = hashlib.sha256(provision_bytes).hexdigest()
    binary = root / 'build-api-arm64/bin/llama-server'
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == config['server_sha256']
    for filename, identity in config['build_files'].items():
        path = root / 'build-api-arm64/bin' / filename
        assert path.stat().st_size == identity['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == identity['sha256']
    listing = subprocess.run([str(binary), '--rpc', config['rpc'], '--list-devices'],
                             capture_output=True, text=True, timeout=20)
    assert listing.returncode == 0 and 'RPC0' in listing.stdout
    result['devices'] = listing.stdout
    d = Path(config['assets']['model_dir'])
    argv = [str(binary), '--rpc', config['rpc'], '-m', str(d / 'Qwen3.8-27B-UD-Q4_K_M.gguf'),
            '--mmproj', str(d / 'mmproj-BF16.gguf'), '--host', '127.0.0.1', '--port', str(port),
            '--alias', 'qwen38-27b-vision-admission', '--ctx-size', '8192', '--parallel', '1',
            '--batch-size', '256', '--ubatch-size', '256', '--gpu-layers', '99',
            '--device', 'MTL0,RPC0', '--mmproj-device', 'MTL0', '--split-mode', 'layer',
            '--tensor-split', '1,1', '--fit', 'off', '--image-min-tokens', '1024',
            '--image-max-tokens', '1024', '--no-warmup', '--no-cont-batching', '--reasoning', 'off',
            '--predict', '96', '--seed', '42', '-lv', '4']
    save('argv.json', argv)
    log = (out / 'server.log').open('xb')
    process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    result['pid'] = process.pid
    save('started.json', result)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 280
    last_observation = 0
    ready = False
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('server exited during loading: ' + str(process.returncode))
        now = time.monotonic()
        if now - last_observation >= 2:
            level = pressure()
            memory = subprocess.run(['ps', '-o', 'pid=,rss=,etime=', '-p', str(process.pid)],
                                    capture_output=True, text=True, timeout=3).stdout
            swap = subprocess.run(['sysctl', 'vm.swapusage'], capture_output=True,
                                  text=True, timeout=3, check=True).stdout.strip()
            result['observations'].append({'timestamp': time.time(), 'pressure': level,
                                           'process_ps_rss_KiB': memory.strip(), 'swap': swap})
            assert level == 1, 'memory pressure rose; aborting capacity test'
            assert (out / 'server.log').stat().st_size < 32 * 1024**2, 'log limit exceeded'
            last_observation = now
        try:
            with opener.open('http://127.0.0.1:' + str(port) + '/health', timeout=1) as response:
                raw = response.read(65537)
                assert len(raw) <= 65536
                health = json.loads(raw)
            if health.get('status') == 'ok':
                save('health.json', health)
                ready = True
                break
        except (OSError, ValueError):
            time.sleep(0.25)
    assert ready, 'startup deadline expired'
    with opener.open('http://127.0.0.1:' + str(port) + '/props', timeout=5) as response:
        raw = response.read(1024**2 + 1)
        assert len(raw) <= 1024**2
        props = json.loads(raw)
    save('props.json', props)
    result['configured_context_observed'] = props.get('default_generation_settings', {}).get('n_ctx')
    assert result['configured_context_observed'] == 8192, 'effective context missing or changed'
    result['postload_pressure'] = pressure()
    assert result['postload_pressure'] == 1
    result['status'] = 'loaded-and-ready-only'
except BaseException as exc:
    result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
finally:
    signal.alarm(0)
    if process is not None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
        result['exit_code'] = process.returncode
        result['owned_process_reaped'] = process.returncode is not None
    if log is not None:
        log.close()
    result['listener_closed'] = closed()
    result['finished_at'] = time.time()
    if (out / 'server.log').exists():
        result['server_log_sha256'] = hashlib.sha256((out / 'server.log').read_bytes()).hexdigest()
    save('result.json', result)
    print(json.dumps(result, indent=2), flush=True)
sys.exit(0 if result['status'] == 'loaded-and-ready-only' and result.get('owned_process_reaped')
         and result['listener_closed'] else 1)
