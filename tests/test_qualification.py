import base64
import hashlib
import json
import os
import socket
import struct
import subprocess
import sys
import threading
import zlib
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def png(width=128, height=128):
    def chunk(kind, body):
        return struct.pack('!I', len(body)) + kind + body + struct.pack('!I', zlib.crc32(kind + body))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!IIBBBBB', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress((b'\0' + b'\xff\0\0' * width) * height)) + chunk(b'IEND', b''))


def manifest(tmp_path):
    image = png()
    (tmp_path / 'red.png').write_bytes(image)
    path = tmp_path / 'cases.json'
    path.write_text(json.dumps({'version': 1, 'cases': [{'id': 'red', 'prompt': 'Name the color.',
                         'image': 'red.png', 'sha256': hashlib.sha256(image).hexdigest()}]}))
    return path


@contextmanager
def server(responder=None, get_responder=None, host='127.0.0.1'):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args):
            pass

        def do_GET(self):
            if get_responder:
                get_responder(self)
            else:
                self.reply(404, b'{}')

        def reply(self, status, body, headers=None):
            self.send_response(status)
            self.send_header('Content-Length', str(len(body)))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            data = self.rfile.read(int(self.headers['Content-Length']))
            requests.append((self.path, data))
            if responder:
                responder(self)
            else:
                self.reply(200, json.dumps({'model': 'control', 'choices': [{'message': {'role': 'assistant', 'content': 'red'}, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 1}}).encode())

    class LocalServer(ThreadingHTTPServer):
        address_family = socket.AF_INET6 if host == '::1' else socket.AF_INET

    httpd = LocalServer((host, 0), Handler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        authority = '[::1]' if host == '::1' else host
        yield f'http://{authority}:{httpd.server_port}', requests
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def invoke(tmp_path, endpoint, source=None, extra=()):
    source = source or manifest(tmp_path)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    return subprocess.run([sys.executable, '-m', 'qualification.client', '--endpoint', endpoint,
                           '--manifest', str(source), '--output', str(tmp_path / 'out'), '--model', 'control',
                           '--timeout', '1', '--deadline', '4', '--max-response-bytes', '4096', *extra],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=10, check=False)


def test_real_image_payload_and_exact_receipt(tmp_path):
    with server() as (endpoint, requests):
        result = invoke(tmp_path, endpoint)
    assert result.returncode == 0, result.stderr
    path, raw = requests[0]
    assert path == '/v1/chat/completions'
    payload = json.loads(raw)
    assert payload['model'] == 'control'
    assert payload['temperature'] == 0
    assert payload['max_tokens'] == 32
    assert payload['stream'] is False
    uri = payload['messages'][0]['content'][1]['image_url']['url']
    assert base64.b64decode(uri.removeprefix('data:image/png;base64,')) == png()
    receipt = json.loads((tmp_path / 'out/receipt.json').read_text())
    assert receipt['success'] is True
    assert receipt['cases'][0]['output'] == 'red'
    assert receipt['cases'][0]['usage']['completion_tokens'] == 1
    assert (tmp_path / 'out' / receipt['cases'][0]['request_body']).read_bytes() == raw
    assert receipt['cases'][0]['response']['status'] == 200


@pytest.mark.parametrize('endpoint', ['http://localhost:1', 'http://127.1:1', 'http://127.0.0.2:1', 'https://127.0.0.1:1', 'http://u@127.0.0.1:1', 'http://127.0.0.1:1/path', 'http://127.0.0.1:1/?q=x', 'http://[::ffff:127.0.0.1]:1'])
def test_endpoint_rejected_before_output(tmp_path, endpoint):
    result = invoke(tmp_path, endpoint)
    assert result.returncode != 0
    assert not (tmp_path / 'out').exists()
    assert 'endpoint' in result.stdout.lower()


def test_exclusive_output_is_never_modified(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    (out / 'receipt.json').write_text('untouchable')
    with server() as (endpoint, requests):
        result = invoke(tmp_path, endpoint)
    assert result.returncode != 0
    assert not requests
    assert (out / 'receipt.json').read_text() == 'untouchable'


@pytest.mark.parametrize('mutation', ['hash', 'duplicate', 'escape', 'symlink', 'empty', 'version', 'crc', 'signature', 'pixels', 'compressed', 'bytes'])
def test_frozen_manifest_validation_before_network(tmp_path, mutation):
    source = manifest(tmp_path)
    data = json.loads(source.read_text())
    extra = []
    if mutation == 'hash':
        data['cases'][0]['sha256'] = '0' * 64
    elif mutation == 'duplicate':
        data['cases'].append(data['cases'][0])
    elif mutation == 'escape':
        escaped = tmp_path.parent / f'{tmp_path.name}.png'
        escaped.write_bytes(png())
        data['cases'][0]['image'] = f'../{escaped.name}'
    elif mutation == 'symlink':
        (tmp_path / 'alias.png').symlink_to(tmp_path / 'red.png')
        data['cases'][0]['image'] = 'alias.png'
    elif mutation == 'empty':
        data['cases'] = []
    elif mutation == 'version':
        data['version'] = 2
    elif mutation == 'bytes':
        extra = ['--max-image-bytes', '12']
    elif mutation == 'pixels':
        extra = ['--max-image-pixels', '100']
    else:
        image = bytearray(png())
        if mutation == 'signature':
            image[0] = 0
        elif mutation == 'crc':
            image[-1] ^= 1
        else:
            image = png(128, 127)
            idat = zlib.compress(b'bad raster')
            chunk = b'IDAT' + idat
            image = image[:33] + struct.pack('!I', len(idat)) + chunk + struct.pack('!I', zlib.crc32(chunk)) + image[-12:]
        (tmp_path / 'red.png').write_bytes(image)
        data['cases'][0]['sha256'] = hashlib.sha256(image).hexdigest()
    source.write_text(json.dumps(data))
    with server() as (endpoint, requests):
        result = invoke(tmp_path, endpoint, source, extra)
    assert result.returncode != 0
    assert not requests
    assert 'error' in result.stdout


@pytest.mark.parametrize('mode', ['redirect', 'oversize', 'chunked', 'broken_chunk', 'timeout', 'trickle', 'bad_json', 'bad_schema', 'http_error'])
def test_failed_http_preserves_bounded_partial_receipt(tmp_path, mode):
    def respond(handler):
        if mode == 'redirect':
            handler.reply(302, b'redirect body', {'Location': 'http://127.0.0.1:1/escape'})
        elif mode in ('oversize', 'http_error'):
            handler.reply(500 if mode == 'http_error' else 200, b'x' * 5000)
        elif mode == 'bad_json':
            handler.reply(200, b'not json')
        elif mode == 'bad_schema':
            handler.reply(200, b'{"choices":[{"message":{"content":null}}]}')
        elif mode in ('chunked', 'broken_chunk'):
            handler.send_response(200)
            handler.send_header('Transfer-Encoding', 'chunked')
            handler.end_headers()
            handler.wfile.write(b'10\r\n' + b'x' * 16 + b'\r\n')
            if mode == 'chunked':
                handler.wfile.write(b'1000\r\n' + b'y' * 4096 + b'\r\n0\r\n\r\n')
            else:
                handler.wfile.write(b'bad-size\r\n')
            handler.wfile.flush()
            handler.close_connection = True
        elif mode == 'trickle':
            handler.send_response(200)
            handler.send_header('Content-Length', '500')
            handler.end_headers()
            try:
                for _ in range(20):
                    handler.wfile.write(b'x')
                    handler.wfile.flush()
                    threading.Event().wait(0.05)
            except OSError:
                pass
        else:
            threading.Event().wait(0.4)
            try:
                handler.reply(200, b'{}')
            except OSError:
                pass
    with server(respond) as (endpoint, requests):
        result = invoke(tmp_path, endpoint, extra=['--timeout', '0.15'])
    assert result.returncode != 0
    receipt = json.loads((tmp_path / 'out/receipt.json').read_text())
    assert receipt['success'] is False
    item = receipt['cases'][0]
    assert item['success'] is False
    assert item['error']
    assert item['response']['bytes'] <= 4096
    assert (tmp_path / 'out' / item['response']['body']).exists()
    assert (tmp_path / 'out' / item['request_wire']).read_bytes().endswith(requests[0][1])


@pytest.mark.parametrize('status', [200, 404, 503])
def test_health_props_observations_not_attestation(tmp_path, status):
    def get(handler):
        handler.reply(status, b'{"model_path":"observed.gguf","status":"ok"}')
    with server(get_responder=get) as (endpoint, requests):
        result = invoke(tmp_path, endpoint)
    receipt = json.loads((tmp_path / 'out/receipt.json').read_text())
    assert receipt['attestation'] is False
    assert receipt['probes'][0]['path'] == '/health'
    assert receipt['probes'][0]['response']['status'] == status
    if status == 503:
        assert result.returncode != 0
        assert not requests
    else:
        assert result.returncode == 0
        assert len(receipt['probes']) == 2
        assert receipt['probes'][1]['supported'] is (status == 200)
        if status == 200:
            assert receipt['probes'][1]['observed']['model_path'] == 'observed.gguf'


@pytest.mark.parametrize('flag,value', [('--timeout', 'nan'), ('--deadline', 'inf'), ('--deadline', '0'), ('--max-response-bytes', '-1'), ('--max-tokens', '0'), ('--max-image-pixels', '999999999999'), ('--max-image-bytes', '0')])
def test_numeric_bounds_rejected_before_output(tmp_path, flag, value):
    result = invoke(tmp_path, 'http://127.0.0.1:1', extra=[flag, value])
    assert result.returncode != 0
    assert not (tmp_path / 'out').exists()
    assert 'bound' in result.stdout.lower()


def test_deadline_spans_requests_and_partial_success_is_failure(tmp_path):
    source = manifest(tmp_path)
    doc = json.loads(source.read_text())
    doc['cases'].append(dict(doc['cases'][0], id='second'))
    source.write_text(json.dumps(doc))
    count = []
    def respond(handler):
        count.append(1)
        if len(count) == 2:
            threading.Event().wait(0.5)
        try:
            handler.reply(200, b'{"model":"control","choices":[{"message":{"role":"assistant","content":"red"},"finish_reason":"stop"}]}')
        except OSError:
            pass
    with server(respond) as (endpoint, _):
        result = invoke(tmp_path, endpoint, source, ['--deadline', '0.2'])
    receipt = json.loads((tmp_path / 'out/receipt.json').read_text())
    assert result.returncode != 0
    assert receipt['success'] is False
    assert receipt['cases'][0]['success'] is True
    assert receipt['cases'][1]['success'] is False
    assert receipt['cases'][1]['elapsed_seconds'] < 0.2
    assert receipt['completed_cases'] == 1
    assert receipt['total_cases'] == 2


@pytest.mark.parametrize('target', ['output_parent', 'manifest_parent', 'output_link', 'output_race'])
def test_symlink_escape_rejected_or_pinned(tmp_path, target):
    source = manifest(tmp_path)
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'sentinel').write_text('unchanged')
    extra = []
    if target == 'output_parent':
        (tmp_path / 'alias').symlink_to(outside, target_is_directory=True)
        extra = ['--output', str(tmp_path / 'alias/out')]
    elif target == 'manifest_parent':
        (tmp_path / 'alias').symlink_to(tmp_path, target_is_directory=True)
        source = tmp_path / 'alias/cases.json'
    elif target == 'output_link':
        (tmp_path / 'out').symlink_to(outside, target_is_directory=True)
    def respond(handler):
        if target == 'output_race':
            (tmp_path / 'out').rename(tmp_path / 'moved')
            (tmp_path / 'out').symlink_to(outside, target_is_directory=True)
        handler.reply(200, b'{"choices":[{"message":{"content":"red"}}]}')
    with server(respond) as (endpoint, requests):
        result = invoke(tmp_path, endpoint, source, extra)
    assert sorted(p.name for p in outside.iterdir()) == ['sentinel']
    if target == 'output_race':
        assert result.returncode == 0
        assert (tmp_path / 'moved/receipt.json').exists()
    else:
        assert result.returncode != 0
        assert not requests


def test_frozen_synthetic_fixtures_are_reproducible(tmp_path):
    result = subprocess.run([sys.executable, '-m', 'qualification.make_fixtures', '--output', str(tmp_path / 'generated')], cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    frozen = ROOT / 'qualification/fixtures'
    doc = json.loads((frozen / 'public-cases.json').read_text())
    assert len(doc['cases']) == 3
    for name in ['public-cases.json', 'red.png', 'blue.png', 'split.png']:
        assert (tmp_path / 'generated' / name).read_bytes() == (frozen / name).read_bytes()
    for case in doc['cases']:
        assert hashlib.sha256((frozen / case['image']).read_bytes()).hexdigest() == case['sha256']
        image = (frozen / case['image']).read_bytes()
        assert struct.unpack('!II', image[16:24]) == (128, 128)
        raster = zlib.decompress(image[41:-16])
        first_row = raster[1:385]
        red, blue = b'\xff\0\0', b'\0\0\xff'
        expected = {'red': red * 128, 'blue': blue * 128, 'split': red * 64 + blue * 64}[case['id']]
        assert first_row == expected
    with server() as (endpoint, requests):
        result = invoke(tmp_path, endpoint, frozen / 'public-cases.json', ['--max-tokens', '1'])
    assert result.returncode == 0, result.stdout
    assert len(requests) == 3
    for _, raw in requests:
        assert json.loads(raw)['max_tokens'] == 1


def test_receipt_checkpoints_bind_inputs_and_requests(tmp_path):
    source = manifest(tmp_path)
    doc = json.loads(source.read_text())
    doc['cases'].append(dict(doc['cases'][0], id='second'))
    source.write_text(json.dumps(doc))
    snapshots = []
    def respond(handler):
        path = tmp_path / 'out/receipt.json'
        snapshots.append(json.loads(path.read_text()) if path.exists() else None)
        handler.reply(200, b'{"model":"observed-control","choices":[{"message":{"role":"assistant","content":"red"},"finish_reason":"stop"}],"usage":{"completion_tokens":1}}')
    with server(respond) as (endpoint, _):
        result = invoke(tmp_path, endpoint, source)
    assert result.returncode == 0
    assert snapshots[0] is not None
    assert snapshots[1]['completed_cases'] == 1
    receipt = json.loads((tmp_path / 'out/receipt.json').read_text())
    assert receipt['manifest_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert receipt['settings']['max_tokens'] == 32
    for item in receipt['cases']:
        assert item['image']['sha256'] == doc['cases'][0]['sha256']
        assert item['image']['width'] == 128
        raw = (tmp_path / 'out' / item['request_body']).read_bytes()
        assert item['request_sha256'] == hashlib.sha256(raw).hexdigest()
        assert item['model_observed'] == 'observed-control'
        assert item['started_at'] <= item['finished_at']
        assert item['response']['complete'] is True


@pytest.mark.parametrize('response', [
    b'{"choices":[{"message":{"content":"red","tool_calls":[{"function":{"name":"exec","arguments":"touch NEVER"}}]}}]}',
    b'{"choices":[{"message":{"content":"red","role":"user"}}]}',
    b'{"choices":[{"message":{"content":"red"}}],"usage":{"completion_tokens":-1}}',
    b'{"choices":[{"message":{"content":"red","content":"blue"}}]}',
    b'{"error":{"message":"failed"},"choices":[{"message":{"content":"red"}}]}',
    b'{"model":42,"choices":[{"message":{"content":"red"}}]}',
])
def test_malformed_or_action_responses_are_not_success(tmp_path, response):
    with server(lambda handler: handler.reply(200, response)) as (endpoint, _):
        result = invoke(tmp_path, endpoint)
    assert result.returncode != 0
    receipt = json.loads((tmp_path / 'out/receipt.json').read_text())
    assert receipt['completed_cases'] == 0
    assert receipt['cases'][0]['success'] is False
    assert (tmp_path / 'out/0000.response.bin').read_bytes() == response
    assert not (ROOT / 'NEVER').exists()


@pytest.mark.parametrize('mode', ['conflicting_lengths', 'ambiguous_framing', 'truncated', 'encoded', 'header_trickle'])
def test_invalid_http_framing_is_failure(tmp_path, mode):
    def respond(handler):
        if mode == 'header_trickle':
            try:
                for piece in [b'HTTP/1.1 200 OK\r\n', b'X-Test: ', b'x', b'x', b'x']:
                    handler.wfile.write(piece)
                    handler.wfile.flush()
                    threading.Event().wait(0.06)
            except OSError:
                pass
        else:
            body = b'{"choices":[{"message":{"content":"red"}}]}'
            headers = b'HTTP/1.1 200 OK\r\nConnection: close\r\n'
            if mode == 'conflicting_lengths':
                headers += f'Content-Length: {len(body)}\r\nContent-Length: 1\r\n'.encode()
            elif mode == 'ambiguous_framing':
                headers += b'Content-Length: 1\r\nTransfer-Encoding: chunked\r\n'
                body = f'{len(body):x}\r\n'.encode() + body + b'\r\n0\r\n\r\n'
            elif mode == 'truncated':
                headers += b'Content-Length: 999\r\n'
            else:
                headers += b'Content-Encoding: gzip\r\n'
            handler.wfile.write(headers + b'\r\n' + body)
        handler.close_connection = True
    with server(respond) as (endpoint, _):
        result = invoke(tmp_path, endpoint, extra=['--timeout', '0.15'])
    assert result.returncode != 0
    receipt = json.loads((tmp_path / 'out/receipt.json').read_text())
    assert receipt['cases'][0]['success'] is False
    assert receipt['cases'][0]['error']


@pytest.mark.parametrize('host', ['127.0.0.1', '::1'])
def test_valid_chunked_json_both_literal_loopbacks(tmp_path, host, monkeypatch):
    monkeypatch.setenv('HTTP_PROXY', 'http://192.0.2.1:9')
    monkeypatch.setenv('ALL_PROXY', 'http://192.0.2.1:9')
    def respond(handler):
        body = b'{"choices":[{"message":{"content":"red"}}]}'
        handler.send_response(200)
        handler.send_header('Transfer-Encoding', 'chunked')
        handler.end_headers()
        for part in (body[:9], body[9:]):
            handler.wfile.write(f'{len(part):x}\r\n'.encode() + part + b'\r\n')
        handler.wfile.write(b'0\r\n\r\n')
        handler.wfile.flush()
    with server(respond, host=host) as (endpoint, requests):
        result = invoke(tmp_path, endpoint)
    assert result.returncode == 0, result.stdout
    assert len(requests) == 1


def test_chunk_data_requires_crlf_retaining_exact_entity(tmp_path):
    body = b'{"choices":[{"message":{"content":"red"}}]}'

    def respond(handler):
        handler.wfile.write(b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n'
                            + f'{len(body):x}\r\n'.encode() + body + b'XX0\r\n\r\n')
        handler.close_connection = True

    with server(respond) as (endpoint, _):
        result = invoke(tmp_path, endpoint)
    assert result.returncode != 0, result.stdout
    receipt = json.loads((tmp_path / 'out/receipt.json').read_text())
    item = receipt['cases'][0]
    assert receipt['success'] is False
    assert receipt['completed_cases'] == 0
    assert item['success'] is False
    assert item['response']['complete'] is False
    assert 'chunk data CRLF' in item['error']
    assert (tmp_path / 'out' / item['response']['body']).read_bytes() == body
    assert item['response']['bytes'] == len(body)
    assert item['response']['sha256'] == hashlib.sha256(body).hexdigest()


def test_last_chunk_requires_complete_empty_trailer_section(tmp_path):
    body = b'{"choices":[{"message":{"content":"red"}}]}'

    def respond(handler):
        handler.wfile.write(b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n'
                            + f'{len(body):x}\r\n'.encode() + body + b'\r\n0\r\n')
        handler.close_connection = True

    with server(respond) as (endpoint, _):
        result = invoke(tmp_path, endpoint)
    assert result.returncode != 0, result.stdout
    receipt = json.loads((tmp_path / 'out/receipt.json').read_text())
    item = receipt['cases'][0]
    assert receipt['success'] is False
    assert receipt['completed_cases'] == 0
    assert item['success'] is False
    assert item['response']['complete'] is False
    assert 'trailer' in item['error']
    assert (tmp_path / 'out' / item['response']['body']).read_bytes() == body
    assert item['response']['bytes'] == len(body)
    assert item['response']['sha256'] == hashlib.sha256(body).hexdigest()


@pytest.mark.parametrize('size_line', [b'+2b\r\n', b'0x2b\r\n', b' 2b\r\n', b'2b\n',
                                       b'2b;foo=bar\r\n', b'2b;bad="unterminated\r\n'])
def test_chunk_size_requires_plain_hex_and_crlf(tmp_path, size_line):
    body = b'{"choices":[{"message":{"content":"red"}}]}'

    def respond(handler):
        handler.wfile.write(b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n'
                            + size_line + body + b'\r\n0\r\n\r\n')
        handler.close_connection = True

    with server(respond) as (endpoint, _):
        result = invoke(tmp_path, endpoint)
    assert result.returncode != 0, result.stdout
    receipt = json.loads((tmp_path / 'out/receipt.json').read_text())
    item = receipt['cases'][0]
    assert receipt['success'] is False
    assert item['response']['complete'] is False
    assert 'chunk size' in item['error']
    assert (tmp_path / 'out' / item['response']['body']).read_bytes() == b''


def test_chunk_size_line_has_separate_byte_bound(tmp_path):
    body = b'{"choices":[{"message":{"content":"red"}}]}'

    def respond(handler):
        size_line = b'0' * 1021 + f'{len(body):x}\r\n'.encode()
        assert len(size_line) == 1025
        handler.wfile.write(b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n'
                            + size_line + body + b'\r\n0\r\n\r\n')
        handler.close_connection = True

    with server(respond) as (endpoint, _):
        result = invoke(tmp_path, endpoint)
    assert result.returncode != 0, result.stdout
    receipt = json.loads((tmp_path / 'out/receipt.json').read_text())
    item = receipt['cases'][0]
    assert item['response']['complete'] is False
    assert 'chunk size line byte limit' in item['error']
    assert (tmp_path / 'out' / item['response']['body']).read_bytes() == b''


def test_chunk_framing_has_aggregate_bound_separate_from_entity(tmp_path):
    body = b'{"choices":[{"message":{"content":"red"}}]}' + b' ' * 57
    size_line = b'0' * 1021 + b'1\r\n'

    def respond(handler):
        chunks = b''.join(size_line + bytes([byte]) + b'\r\n' for byte in body)
        handler.wfile.write(b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n'
                            + chunks + b'0\r\n\r\n')
        handler.close_connection = True

    with server(respond) as (endpoint, _):
        result = invoke(tmp_path, endpoint)
    assert result.returncode != 0, result.stdout
    receipt = json.loads((tmp_path / 'out/receipt.json').read_text())
    item = receipt['cases'][0]
    expected = body[:65536 // (len(size_line) + 2)]
    assert receipt['success'] is False
    assert item['response']['complete'] is False
    assert 'chunk framing byte limit' in item['error']
    assert (tmp_path / 'out' / item['response']['body']).read_bytes() == expected
    assert item['response']['bytes'] == len(expected)
    assert item['response']['sha256'] == hashlib.sha256(expected).hexdigest()


@pytest.mark.parametrize('framing_bytes', [65536, 65537])
def test_chunk_framing_budget_includes_terminal_crlf(tmp_path, framing_bytes):
    body = b'{"choices":[{"message":{"content":"red"}}]}' + b' ' * 57
    padding = framing_bytes - len(body) * 5 - 6
    chunks = []
    for byte in body:
        zeros = min(1021, padding)
        chunks.append(b'0' * zeros + b'1\r\n' + bytes([byte]) + b'\r\n')
        padding -= zeros
    assert padding == 0
    framed = b''.join(chunks) + b'00\r\n\r\n'
    assert len(framed) - len(body) == framing_bytes

    def respond(handler):
        handler.wfile.write(b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n'
                            + framed)
        handler.close_connection = True

    with server(respond) as (endpoint, _):
        result = invoke(tmp_path, endpoint)
    receipt = json.loads((tmp_path / 'out/receipt.json').read_text())
    item = receipt['cases'][0]
    succeeds = framing_bytes == 65536
    assert (result.returncode == 0) is succeeds, result.stdout
    assert receipt['success'] is succeeds
    assert item['response']['complete'] is succeeds
    assert (tmp_path / 'out' / item['response']['body']).read_bytes() == body
    if not succeeds:
        assert 'chunk framing byte limit' in item['error']


@pytest.mark.parametrize('tail', [b'\r\n0\r\n\n', b'\r\n0\r\n\r',
                                 b'\r\n0\r\nX-Test: yes\r\n\r\n',
                                 b'\r\n0\r\n' + b'X' * 100000,
                                 b'\r\n0;foo=bar\r\n\r\n', b'\n0\r\n\r\n'],
                         ids=['terminal_lf', 'terminal_cr', 'trailer', 'large_trailer',
                              'terminal_extension', 'data_lf'])
def test_unsupported_chunk_trailers_and_partial_delimiters_fail(tmp_path, tail):
    body = b'{"choices":[{"message":{"content":"red"}}]}'

    def respond(handler):
        handler.wfile.write(b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n'
                            + f'{len(body):X}\r\n'.encode() + body + tail)
        handler.close_connection = True

    with server(respond) as (endpoint, _):
        result = invoke(tmp_path, endpoint)
    assert result.returncode != 0, result.stdout
    receipt = json.loads((tmp_path / 'out/receipt.json').read_text())
    item = receipt['cases'][0]
    assert receipt['success'] is False
    assert item['response']['complete'] is False
    assert (tmp_path / 'out' / item['response']['body']).read_bytes() == body
    assert item['response']['sha256'] == hashlib.sha256(body).hexdigest()


@pytest.mark.parametrize('phase', ['size', 'trailer'])
def test_chunk_framing_reads_keep_absolute_watchdog(tmp_path, phase):
    body = b'{"choices":[{"message":{"content":"red"}}]}'

    def respond(handler):
        try:
            handler.wfile.write(b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n')
            if phase == 'trailer':
                handler.wfile.write(f'{len(body):x}\r\n'.encode() + body + b'\r\n0\r\n\r')
                handler.wfile.flush()
                threading.Event().wait(0.4)
            else:
                for _ in range(20):
                    handler.wfile.write(b'0')
                    handler.wfile.flush()
                    threading.Event().wait(0.03)
        except OSError:
            pass
        handler.close_connection = True

    with server(respond) as (endpoint, _):
        result = invoke(tmp_path, endpoint, extra=['--timeout', '0.15'])
    assert result.returncode != 0, result.stdout
    receipt = json.loads((tmp_path / 'out/receipt.json').read_text())
    item = receipt['cases'][0]
    assert item['response']['complete'] is False
    assert 'TimeoutError' in item['error']
    assert item['elapsed_seconds'] < 0.6
    expected = body if phase == 'trailer' else b''
    assert (tmp_path / 'out' / item['response']['body']).read_bytes() == expected


@pytest.mark.parametrize('mode', ['exact_cap', 'over_cap', 'truncated_data'])
def test_chunked_entity_prefix_and_body_cap_are_unchanged(tmp_path, mode):
    body = b'{"choices":[{"message":{"content":"red"}}]}'

    def respond(handler):
        sent = body + (b' ' if mode == 'over_cap' else b'')
        advertised = len(sent) + (10 if mode == 'truncated_data' else 0)
        tail = b'' if mode == 'truncated_data' else b'\r\n00\r\n\r\n'
        handler.wfile.write(b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n'
                            + f'{advertised:X}\r\n'.encode() + sent + tail)
        handler.close_connection = True

    with server(respond) as (endpoint, _):
        result = invoke(tmp_path, endpoint, extra=['--max-response-bytes', str(len(body))])
    receipt = json.loads((tmp_path / 'out/receipt.json').read_text())
    item = receipt['cases'][0]
    succeeds = mode == 'exact_cap'
    assert (result.returncode == 0) is succeeds, result.stdout
    assert receipt['success'] is succeeds
    assert item['response']['complete'] is succeeds
    assert (tmp_path / 'out' / item['response']['body']).read_bytes() == body
    assert item['response']['bytes'] == len(body)
    assert item['response']['sha256'] == hashlib.sha256(body).hexdigest()
    if mode == 'over_cap':
        assert 'response byte limit' in item['error']
    elif mode == 'truncated_data':
        assert 'IncompleteRead' in item['error']


def test_invalid_unicode_response_retains_error_receipt(tmp_path):
    response = b'{"choices":[{"message":{"content":"\\ud800"}}]}'
    with server(lambda handler: handler.reply(200, response)) as (endpoint, _):
        result = invoke(tmp_path, endpoint)
    assert result.returncode != 0
    receipt = json.loads((tmp_path / 'out/receipt.json').read_text())
    item = receipt['cases'][0]
    assert item['success'] is False
    assert item['response']['status'] == 200
    assert item['error']
    assert receipt['probes'][0]['started_at'] <= receipt['probes'][0]['finished_at']
    wire = (tmp_path / 'out' / item['request_wire']).read_bytes()
    assert item['request_wire_sha256'] == hashlib.sha256(wire).hexdigest()
