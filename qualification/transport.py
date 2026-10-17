"""No proxies, redirects, retries, decompression, or action execution."""
import hashlib
import http.client
import socket
import threading
import time
from datetime import datetime, timezone


class StrictChunkResponse(http.client.HTTPResponse):
    """Keep stdlib HTTP/body reads, but validate chunk framing at its seam."""

    _framing_bytes = 0

    def _read_framing(self, amount, *, line=False):
        # One sentinel byte detects overflow without reading arbitrary framing.
        reader = self.fp.readline if line else self.fp.read
        data = reader(min(amount, 65536 - self._framing_bytes + 1))
        self._framing_bytes += len(data)
        if self._framing_bytes > 65536:
            raise ValueError('chunk framing byte limit exceeded')
        return data

    def _read_next_chunk_size(self):
        line = self._read_framing(1025, line=True)
        if len(line) > 1024:
            raise ValueError('chunk size line byte limit exceeded')
        size = line[:-2]
        if (not line.endswith(b'\r\n') or not size
                or any(byte not in b'0123456789abcdefABCDEF' for byte in size)):
            raise ValueError('unsupported or malformed chunk size; plain hex and CRLF required')
        return int(size, 16)

    def _read_and_discard_trailer(self):
        if self._read_framing(2) != b'\r\n':
            raise ValueError('missing trailer-section CRLF or unsupported trailers')

    def _get_chunk_left(self):
        chunk_left = self.chunk_left
        if not chunk_left:
            if chunk_left is not None and self._read_framing(2) != b'\r\n':
                raise ValueError('invalid or missing chunk data CRLF')
            chunk_left = self._read_next_chunk_size()
            if chunk_left == 0:
                self._read_and_discard_trailer()
                self._close_conn()
                chunk_left = None
            self.chunk_left = chunk_left
        return chunk_left


def exchange(endpoint, method, path, body, timeout, deadline, limit, output, prefix):
    started = time.monotonic()
    budget = min(timeout, deadline - started)
    record = {'success': False, 'method': method, 'path': path,
              'started_at': datetime.now(timezone.utc).isoformat(),
              'request_wire': f'{prefix}.request.http',
              'response': {'status': None, 'headers': [], 'body': f'{prefix}.response.bin', 'bytes': 0, 'complete': False}}
    data = bytearray()
    wire = bytearray()
    timer = None
    expired = threading.Event()

    class RecordingConnection(http.client.HTTPConnection):
        response_class = StrictChunkResponse

        def send(self, data):
            wire.extend(data)
            super().send(data)

    connection = RecordingConnection(endpoint.hostname, endpoint.port, timeout=max(0.001, budget))
    try:
        if budget <= 0:
            raise TimeoutError('run deadline exceeded before request')
        connection.connect()
        sock = connection.sock

        def expire():
            expired.set()
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        remaining = budget - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError('request time budget exceeded connecting')
        timer = threading.Timer(remaining, expire)
        timer.daemon = True
        timer.start()
        headers = {'Connection': 'close', 'Accept': 'application/json', 'Accept-Encoding': 'identity'}
        if body is not None:
            headers['Content-Type'] = 'application/json'
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        record['response'].update(status=response.status, headers=response.getheaders())
        lengths = response.headers.get_all('Content-Length', [])
        transfer = response.headers.get_all('Transfer-Encoding', [])
        if (len(lengths) > 1 or (lengths and transfer)
                or (lengths and (not lengths[0].isascii() or not lengths[0].isdigit()))
                or (transfer and transfer != ['chunked'])):
            raise ValueError('ambiguous or unsupported HTTP response framing')
        encodings = response.headers.get_all('Content-Encoding', [])
        if encodings and encodings != ['identity']:
            raise ValueError('encoded HTTP bodies are unsupported')
        while True:
            try:
                piece = response.read1(min(8192, limit + 1 - len(data)))
            except http.client.IncompleteRead as exc:
                data.extend(exc.partial[:limit - len(data)])
                raise
            if not piece:
                break
            if len(data) + len(piece) > limit:
                data.extend(piece[:limit - len(data)])
                raise ValueError('response byte limit exceeded; retained bounded prefix')
            data.extend(piece)
        if expired.is_set() or time.monotonic() - started >= budget:
            raise TimeoutError('request time budget exceeded')
        if response.length not in (None, 0):
            raise ValueError('truncated Content-Length response')
        record['response']['complete'] = True
        if not 200 <= response.status < 300:
            raise ValueError(f'HTTP status {response.status}; redirects are never followed')
        record['success'] = True
    except Exception as exc:  # noqa: BLE001 - persist any transport failure, never count it as success
        record['error'] = ('TimeoutError: request time budget exceeded' if expired.is_set()
                           else f'{type(exc).__name__}: {exc}')
    finally:
        if timer:
            timer.cancel()
            timer.join()
        connection.close()
        output.write(record['request_wire'], bytes(wire))
        output.write(record['response']['body'], bytes(data))
        record['response'].update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        record['request_wire_sha256'] = hashlib.sha256(wire).hexdigest()
        record['finished_at'] = datetime.now(timezone.utc).isoformat()
        record['elapsed_seconds'] = time.monotonic() - started
    return record, bytes(data)
