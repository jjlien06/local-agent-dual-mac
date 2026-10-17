"""Owned, finite one-shot loading diagnostic; not a service launcher."""
import errno
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time

root = Path('/Users/jeremylien/Projects/local-agent-dual-mac')
source = root / 'results/full-target-admission/load-remote-v3.py'
local_ip = sys.argv[1]
assert local_ip.startswith('169.254.')
rpc_port = 18574
out = root / 'results/full-target-load-v3'
out.mkdir(exist_ok=False)
rpc = None
ssh = None
rpc_log = None
remote_log = None
result = {'role': 'full-target-load-only', 'status': 'failed', 'observations': [],
          'started_at': time.time(), 'inference_executed': False}

def save(name, obj):
    with (out / name).open('x') as stream:
        json.dump(obj, stream, indent=2)

def pressure():
    return int(subprocess.run(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'],
                             capture_output=True, text=True, timeout=3, check=True).stdout)

def owned_footprint(pid):
    probe = subprocess.run(['/usr/bin/footprint', '-p', str(pid), '-f', 'bytes', '--noCategories'], capture_output=True, text=True, timeout=8)
    return {'exit_code': probe.returncode, 'stdout': probe.stdout[:32768], 'stderr': probe.stderr[:4096]}

def closed():
    with socket.socket() as sock:
        sock.settimeout(1.5)
        return sock.connect_ex((local_ip, rpc_port)) == errno.ECONNREFUSED

def stop(signum, frame):
    raise TimeoutError('bounded controller interrupted: ' + str(signum))

for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGALRM):
    signal.signal(sig, stop)
signal.alarm(440)
try:
    assert closed() and pressure() == 1
    binary = root / 'build-api-arm64/bin/ggml-rpc-server'
    local_build = json.loads((root / 'results/full-target-admission/laptop-build-snapshot.json').read_text())
    for filename, identity in local_build['files'].items():
        path = binary.parent / filename
        assert path.stat().st_size == identity['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == identity['sha256']
    mini_build = json.loads((root / 'results/full-target-admission/mini-build-snapshot.json').read_text())
    config = {'role': 'full-target-load-only', 'port': 18406, 'rpc': local_ip + ':' + str(rpc_port),
              'source_revision': mini_build['source_revision'],
              'assets': json.loads((root / 'full-target-assets.json').read_text()),
              'server_sha256': mini_build['files']['llama-server']['sha256'],
              'build_files': mini_build['files'],
              'remote_supervisor_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    save('admitted-config.json', config)
    argv = [str(binary), '--host', local_ip, '--port', str(rpc_port), '--device', 'MTL0']
    save('rpc-argv.json', argv)
    env = dict(os.environ, GGML_RPC_NO_RDMA='1', GGML_RPC_DEBUG='3', GGML_METAL_NO_RESIDENCY='1', TMPDIR='/Users/jeremylien/.hermes/cache/scratch')
    result['rpc_residency_policy'] = {'GGML_METAL_NO_RESIDENCY': env['GGML_METAL_NO_RESIDENCY']}
    rpc_log = (out / 'local-rpc.log').open('xb')
    rpc = subprocess.Popen(argv, stdout=rpc_log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    result['rpc_pid'] = rpc.pid
    save('rpc-started.json', {'pid': rpc.pid, 'timestamp': time.time()})
    listener = None
    startup_deadline = time.monotonic() + 15
    while time.monotonic() < startup_deadline:
        assert rpc.poll() is None, 'RPC exited before listening'
        q = subprocess.run(['lsof', '-nP', '-a', '-p', str(rpc.pid), '-iTCP:' + str(rpc_port),
                            '-sTCP:LISTEN'], capture_output=True, text=True, timeout=3)
        if q.returncode == 0:
            listener = q.stdout
            break
        time.sleep(0.2)
    assert listener and local_ip in listener
    result['owned_listener'] = listener
    params = json.loads(Path('/Users/jeremylien/Projects/paretoquant-metal/results/qwen38-speed-v2/fast-profile-v2.json').read_text())['parameters']
    command = 'TMPDIR=/Users/jeremylien/.hermes/cache/scratch GGML_RPC_NO_RDMA=1 ' + shlex.join([
        '/Users/jeremylien/Projects/local-agent-dual-mac/.build-venv/bin/python', '-u', '-c', source.read_text()])
    argv = ['ssh', '-T', '-i', params['ssh_key'], '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes',
            '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=8',
            '-o', 'HostKeyAlias=' + params['ssh_host_key_alias'], params['ssh_target'], command]
    remote_log = (out / 'remote-controller.log').open('xb')
    ssh = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=remote_log, stderr=subprocess.STDOUT, env=env)
    ssh.stdin.write((json.dumps(config) + '\n').encode())
    ssh.stdin.close()
    deadline = time.monotonic() + 375
    guard_error = None
    while ssh.poll() is None:
        if time.monotonic() >= deadline:
            raise TimeoutError('remote supervisor did not return within admitted lifetime')
        level = pressure()
        rss = subprocess.run(['ps', '-o', 'pid=,rss=,etime=', '-p', str(rpc.pid)],
                             capture_output=True, text=True, timeout=3).stdout.strip()
        swap = subprocess.run(['sysctl', 'vm.swapusage'], capture_output=True,
                              text=True, timeout=3, check=True).stdout.strip()
        result['observations'].append({'timestamp': time.time(), 'pressure': level,
                                       'process_ps_rss_KiB': rss, 'swap': swap, 'owned_rpc_footprint': owned_footprint(rpc.pid), 'global_vm_stat': subprocess.run(['vm_stat'], capture_output=True, text=True, timeout=5).stdout})
        if level != 1 or (out / 'local-rpc.log').stat().st_size >= 32 * 1024**2:
            guard_error = 'local pressure or log guard triggered'
            if rpc.poll() is None:
                rpc.terminate()
                rpc.wait(timeout=15)
        time.sleep(2)
    result['remote_controller_exit'] = ssh.returncode
    assert guard_error is None, guard_error
    assert ssh.returncode == 0, 'remote load diagnostic failed; preserved remote log'
    result['status'] = 'remote-load-diagnostic-returned-success'
except BaseException as exc:
    result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
finally:
    signal.alarm(0)
    if ssh is not None and ssh.poll() is None:
        # The remote supervisor has its own 330-second alarm and cleanup.
        ssh.terminate()
        try:
            ssh.wait(timeout=10)
        except subprocess.TimeoutExpired:
            ssh.kill()
            ssh.wait(timeout=10)
        result['remote_cleanup_needs_readback'] = True
    if rpc is not None:
        if rpc.poll() is None:
            rpc.terminate()
            try:
                rpc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                rpc.kill()
                rpc.wait(timeout=15)
        result['owned_rpc_reaped'] = rpc.returncode is not None
        result['rpc_exit'] = rpc.returncode
    if rpc_log is not None:
        rpc_log.close()
    if remote_log is not None:
        remote_log.close()
    result['rpc_listener_closed'] = closed()
    result['finished_at'] = time.time()
    for name in ('local-rpc.log', 'remote-controller.log'):
        if (out / name).exists():
            result[name + '_sha256'] = hashlib.sha256((out / name).read_bytes()).hexdigest()
    save('local-result.json', result)
    print(json.dumps(result, indent=2), flush=True)
sys.exit(0 if result['status'] == 'remote-load-diagnostic-returned-success'
         and result.get('owned_rpc_reaped') and result['rpc_listener_closed'] else 1)
