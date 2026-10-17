"""Bounded loopback multimodal qualification; never executes model output."""
import argparse
import base64
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from qualification.inputs import load_manifest, parse_json
from qualification.store import Output
from qualification.transport import exchange


def stamp():
    return datetime.now(timezone.utc).isoformat()


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--endpoint', required=True, help='HTTP origin at literal 127.0.0.1 or [::1]; no path, auth, or redirects')
    parser.add_argument('--manifest', required=True, type=Path, help='Existing frozen version-1 public cases JSON (relative PNG paths and exact SHA-256)')
    parser.add_argument('--output', required=True, type=Path, help='NEW exclusive output directory; parent must already exist without symlinks')
    parser.add_argument('--model', required=True, help='Explicit server model ID/alias; recorded, not attested')
    parser.add_argument('--timeout', type=float, default=30, help='Absolute time budget in seconds per HTTP exchange, including trickle traffic (max 3600)')
    parser.add_argument('--deadline', type=float, default=300, help='Shared run budget in seconds from argument parsing; no new network work after expiry (max 86400)')
    parser.add_argument('--max-response-bytes', type=int, default=1048576, help='Maximum decoded HTTP entity-body bytes retained per exchange (max 16777216)')
    parser.add_argument('--max-tokens', type=int, default=32, help='Explicit completion budget; use 1 for a separate first-token control (max 32768)')
    parser.add_argument('--max-image-bytes', type=int, default=4194304, help='Maximum bytes per PNG (max 16777216)')
    parser.add_argument('--max-image-pixels', type=int, default=1048576, help='Maximum width*height per PNG (max 16777216)')
    args = parser.parse_args(argv)
    start = time.monotonic()
    receipt = {'success': False, 'started_at': stamp(), 'cases': [], 'model_requested': args.model,
               'attestation': False, 'probes': [], 'endpoint': args.endpoint}
    output = None

    def checkpoint():
        receipt['completed_cases'] = sum(item.get('success') is True for item in receipt['cases'])
        receipt['elapsed_seconds'] = time.monotonic() - start
        if output:
            output.write('receipt.json', encode(receipt), replace=True)

    try:
        for key, cap in [('timeout', 3600), ('deadline', 86400), ('max_response_bytes', 16777216),
                         ('max_tokens', 32768), ('max_image_bytes', 16777216), ('max_image_pixels', 16777216)]:
            value = getattr(args, key)
            if not math.isfinite(value) or not 0 < value <= cap:
                raise ValueError(f'{key} bound must be finite and in (0, {cap}]')
        if not args.model or len(args.model) > 1024:
            raise ValueError('model identifier bound is 1..1024 characters')
        endpoint = urlsplit(args.endpoint)
        expected_authority = (f'[{endpoint.hostname}]' if endpoint.hostname == '::1' else endpoint.hostname)
        if endpoint.port is not None:
            expected_authority = f'{expected_authority}:{endpoint.port}'
        if (endpoint.scheme != 'http' or endpoint.hostname not in ('127.0.0.1', '::1')
                or endpoint.netloc != expected_authority or endpoint.path not in ('', '/')
                or endpoint.query or endpoint.fragment or endpoint.port == 0):
            raise ValueError('endpoint must be a literal HTTP loopback origin')
        source, cases = load_manifest(args.manifest, args.max_image_bytes, args.max_image_pixels)
        receipt['total_cases'] = len(cases)
        receipt['settings'] = {key: getattr(args, key) for key in (
            'timeout', 'deadline', 'max_response_bytes', 'max_tokens', 'max_image_bytes', 'max_image_pixels')}
        receipt['settings'].update(temperature=0, stream=False)
        output = Output(args.output)
        output.write('manifest.json', source)
        receipt['manifest_sha256'] = hashlib.sha256(source).hexdigest()
        checkpoint()
        for name in ('health', 'props'):
            probe, body = exchange(endpoint, 'GET', f'/{name}', None, args.timeout,
                                   start + args.deadline, args.max_response_bytes, output, name)
            receipt['probes'].append(probe)
            probe['supported'] = probe['response']['status'] not in (404, 405, 501)
            if not probe['supported'] and probe['response']['complete']:
                checkpoint()
                continue
            if not probe['success']:
                raise ValueError(probe['error'])
            try:
                probe['observed'] = parse_json(body)
            except Exception as exc:
                probe['error'] = f'{type(exc).__name__}: {exc}'
                probe['success'] = False
                raise
            checkpoint()
        for number, (case, image, info) in enumerate(cases):
            payload = {'model': args.model, 'temperature': 0, 'max_tokens': args.max_tokens, 'stream': False,
                       'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': case['prompt']},
                          {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(image).decode('ascii')}}]}]}
            raw = encode(payload)
            item = {'id': case['id'], 'request_body': f'{number:04d}.request.json', 'started_at': stamp(),
                    'success': False, 'request_sha256': hashlib.sha256(raw).hexdigest(),
                    'image': dict(info, sha256=case['sha256'], path=case['image'])}
            receipt['cases'].append(item)
            output.write(item['request_body'], raw)
            checkpoint()
            observation, body = exchange(endpoint, 'POST', '/v1/chat/completions', raw, args.timeout,
                                         start + args.deadline, args.max_response_bytes, output, f'{number:04d}')
            item.update(observation)
            if not item['success']:
                raise ValueError(item['error'])
            item['success'] = False
            try:
                parsed = parse_json(body)
                if not isinstance(parsed, dict) or 'error' in parsed:
                    raise ValueError('response must be a completion object without an error')
                choices = parsed['choices']
                if not isinstance(choices, list) or len(choices) != 1:
                    raise ValueError('response must have exactly one choice')
                message = choices[0]['message']
                if message.get('role', 'assistant') != 'assistant' or message.get('tool_calls') or message.get('function_call'):
                    raise ValueError('response must be assistant text without action calls')
                content = message['content']
                if not isinstance(content, str) or not content.strip():
                    raise ValueError('response requires nonempty text content')
                usage = parsed.get('usage')
                if usage is not None:
                    if not isinstance(usage, dict):
                        raise ValueError('usage must be an object or null')
                    for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                        if field in usage and (type(usage[field]) is not int or usage[field] < 0):
                            raise ValueError('usage token counts must be nonnegative integers')
                model = parsed.get('model')
                if model is not None and (not isinstance(model, str) or not model):
                    raise ValueError('observed model identity must be a nonempty string or null')
                item.update(output=content, usage=parsed.get('usage'), model_observed=parsed.get('model'),
                            finish_reason=choices[0].get('finish_reason'), success=True)
            except Exception as exc:
                item['error'] = f'{type(exc).__name__}: {exc}'
                raise
            finally:
                item['finished_at'] = stamp()
            checkpoint()
        receipt['elapsed_seconds'] = time.monotonic() - start
        receipt['success'] = True
    except Exception as exc:  # noqa: BLE001 - fail closed while retaining a partial receipt
        receipt['error'] = f'{type(exc).__name__}: {exc}'
        if receipt['cases'] and not receipt['cases'][-1].get('success'):
            receipt['cases'][-1].setdefault('error', receipt['error'])
            receipt['cases'][-1]['finished_at'] = stamp()
    receipt['finished_at'] = stamp()
    checkpoint()
    if output:
        output.close()
    print(encode(receipt).decode())
    return 0 if receipt['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
