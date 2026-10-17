"""Read-only CPU verification of existing exact-27B MLX vision checkpoint."""
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time

raw = sys.stdin.buffer.read(2 * 1024**2 + 1)
if len(raw) > 2 * 1024**2:
    raise ValueError('admission bound')
x = json.loads(raw)
if x['schema'] != 'exact-mlx-vision-readonly-admission-v1':
    raise ValueError('wrong admission schema')
if x['revision'] != '10c35caafbb80f7dc6a7a432cdd11af10a6d4818':
    raise ValueError('wrong checkpoint identity')
root = Path(x['checkpoint_dir'])
project = Path('/Users/jeremylien/Projects/local-agent-dual-mac')
if root.is_symlink() or root.resolve() != root:
    raise ValueError('checkpoint root must be canonical and nonsymlink')
out = project / 'results/mlx-vision-readback-v1'
out.mkdir(exist_ok=False)
receipt = {'schema': 'exact-mlx-vision-readback-v1', 'status': 'failed', 'repo': x['repo'],
           'revision': x['revision'], 'started_at': time.time(), 'shards': [], 'vision': {},
           'metadata': {}, 'observations': [], 'gpu_execution': False, 'model_imported': False,
           'checkpoint_modified': False, 'admission_sha256': hashlib.sha256(raw).hexdigest()}

def pressure():
    value = int(subprocess.run(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'],
                               capture_output=True, text=True, check=True, timeout=4).stdout)
    receipt['observations'].append({'timestamp': time.time(), 'pressure': value})
    if value != 1:
        raise RuntimeError('normal-pressure prerequisite failed')

def signature(stat):
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]

def stop(signum, frame):
    raise TimeoutError('read-only verification deadline/signal ' + str(signum))

for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGHUP):
    signal.signal(sig, stop)
signal.alarm(180)
try:
    pressure()
    for name, expected in x['metadata_sha256'].items():
        if Path(name).name != name:
            raise ValueError('metadata path')
        path = root / name
        if path.is_symlink():
            raise ValueError('metadata symlink')
        data = path.read_bytes()
        if len(data) > 16 * 1024**2 or hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('metadata digest mismatch: ' + name)
        receipt['metadata'][name] = {'bytes': len(data), 'sha256': expected}
    all_names = set()
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    for entry in x['shards']:
        name = entry['name']
        if Path(name).name != name or name not in ('model-00001-of-00003.safetensors', 'model-00002-of-00003.safetensors', 'model-00003-of-00003.safetensors'):
            raise ValueError('shard path')
        path = root / name
        if path.is_symlink():
            raise ValueError('shard symlink')
        with path.open('rb') as stream:
            initial = os.fstat(stream.fileno())
            if initial.st_size != entry['bytes']:
                raise ValueError('shard size mismatch')
            digest = hashlib.sha256()
            next_pressure_at = 512 * 1024**2
            count = 0
            while True:
                block = stream.read(8 * 1024**2)
                if not block:
                    break
                digest.update(block)
                count += len(block)
                if count >= next_pressure_at:
                    pressure()
                    next_pressure_at += 512 * 1024**2
            if count != entry['bytes'] or digest.hexdigest() != entry['sha256']:
                raise ValueError('publisher whole-shard digest mismatch: ' + name)
            stream.seek(0)
            length_bytes = stream.read(8)
            if len(length_bytes) != 8:
                raise ValueError('truncated header length')
            length = struct.unpack('<Q', length_bytes)[0]
            if not 0 < length <= 2 * 1024**2:
                raise ValueError('header length bound')
            header_raw = stream.read(length)
            header = json.loads(header_raw)
            if header != entry['header']:
                raise ValueError('header differs from trusted pinned inventory: ' + name)
            for tensor, descriptor in header.items():
                if tensor == '__metadata__':
                    continue
                if tensor in all_names or index.get(tensor) != name:
                    raise ValueError('duplicate/unexpected/index tensor: ' + tensor)
                all_names.add(tensor)
                if not tensor.startswith('vision_tower.'):
                    continue
                begin, end = descriptor['data_offsets']
                size = end - begin
                if (descriptor['dtype'] != 'BF16' or size != math.prod(descriptor['shape']) * 2
                        or begin < 0 or end < begin or 8 + length + end > initial.st_size):
                    raise ValueError('vision dtype/shape/offset contract: ' + tensor)
                stream.seek(8 + length + begin)
                h = hashlib.sha256()
                remaining = size
                while remaining:
                    block = stream.read(min(8 * 1024**2, remaining))
                    if not block:
                        raise ValueError('truncated vision tensor: ' + tensor)
                    h.update(block)
                    remaining -= len(block)
                receipt['vision'][tensor] = {'file': name, 'shape': descriptor['shape'],
                                             'dtype': descriptor['dtype'], 'data_offsets': [begin, end],
                                             'bytes': size, 'sha256': h.hexdigest()}
            final = os.fstat(stream.fileno())
            if signature(final) != signature(initial) or signature(path.stat()) != signature(initial):
                raise ValueError('shard changed during readback: ' + name)
            receipt['shards'].append({'file': name, 'bytes': count, 'sha256': digest.hexdigest(),
                                      'header_sha256': hashlib.sha256(header_raw).hexdigest(),
                                      'stat_signature': signature(initial)})
            pressure()
    if all_names != set(index) or len(all_names) != x['expected_full_tensor_count']:
        raise ValueError('full inventory coverage')
    if len(receipt['vision']) != x['expected_vision_tensor_count']:
        raise ValueError('vision inventory coverage')
    vision_bytes = sum(item['bytes'] for item in receipt['vision'].values())
    if vision_bytes != x['expected_vision_payload_bytes']:
        raise ValueError('vision payload byte count')
    config = json.loads((root / 'config.json').read_text())
    if config['vision_config'].get('deepstack_visual_indexes') != []:
        raise ValueError('unexpected DeepStack')
    if receipt['vision']['vision_tower.patch_embed.proj.weight']['shape'] != [1152, 2, 16, 16, 3]:
        raise ValueError('already-converted convolution layout changed')
    receipt.update(status='whole-shards-and-vision-payloads-verified', full_tensor_count=len(all_names),
                   text_tensor_count=len(all_names) - len(receipt['vision']),
                   vision_tensor_count=len(receipt['vision']), vision_payload_bytes=vision_bytes)
except BaseException as exc:
    receipt['error'] = {'type': type(exc).__name__, 'message': str(exc)}
finally:
    signal.alarm(0)
    receipt['finished_at'] = time.time()
    with (out / 'result.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps({'status': receipt['status'], 'error': receipt.get('error'),
                      'full_tensor_count': receipt.get('full_tensor_count'),
                      'vision_tensor_count': len(receipt['vision']),
                      'vision_payload_bytes': receipt.get('vision_payload_bytes'),
                      'shards_verified': len(receipt['shards']),
                      'result_sha256': hashlib.sha256((out / 'result.json').read_bytes()).hexdigest(),
                      'result_path': str(out / 'result.json'), 'gpu_execution': False}, indent=2), flush=True)
sys.exit(0 if receipt['status'] == 'whole-shards-and-vision-payloads-verified' else 1)
