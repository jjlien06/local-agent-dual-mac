"""Strict, bounded frozen RGB/RGBA PNG case inputs (standard library only)."""
import hashlib
import json
import os
import re
import stat
import struct
import zlib
from pathlib import Path


def safe_open(path, flags=os.O_RDONLY):
    """Walk each path component without following symlinks (POSIX)."""
    path = Path(os.path.abspath(path))
    if path == Path('/'):
        return os.open('/', flags | os.O_NOFOLLOW | os.O_NONBLOCK)
    directory = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        return os.open(path.name, flags | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)


def read_bounded(path, limit):
    with os.fdopen(safe_open(path), 'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('input is not a regular file')
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError('input byte limit exceeded')
    return data


def parse_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f'nonfinite JSON number: {value}')
    result = json.loads(data.decode('utf-8'), object_pairs_hook=pairs, parse_constant=constant)
    # Reject escaped lone surrogates and numeric overflow before storing a receipt.
    json.dumps(result, ensure_ascii=False, allow_nan=False).encode('utf-8')
    return result


def validate_png(data, max_pixels):
    if data[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('PNG signature mismatch')
    offset, chunks, compressed = 8, [], bytearray()
    width = height = channels = None
    while offset < len(data):
        if offset + 12 > len(data):
            raise ValueError('truncated PNG chunk')
        size = struct.unpack_from('!I', data, offset)[0]
        end = offset + 12 + size
        if end > len(data):
            raise ValueError('truncated PNG data')
        kind = data[offset + 4:offset + 8]
        body = data[offset + 8:end - 4]
        crc = struct.unpack_from('!I', data, end - 4)[0]
        if zlib.crc32(kind + body) != crc:
            raise ValueError('PNG CRC mismatch')
        if not chunks and kind != b'IHDR':
            raise ValueError('PNG must start with IHDR')
        if kind == b'IHDR':
            if chunks or size != 13:
                raise ValueError('invalid PNG IHDR')
            width, height, depth, color, compression, filtering, interlace = struct.unpack('!IIBBBBB', body)
            if (not width or not height or width * height > max_pixels
                    or depth != 8 or color not in (2, 6) or compression or filtering or interlace):
                raise ValueError('PNG dimensions/pixel limit or unsupported format (RGB/RGBA8 noninterlaced only)')
            channels = 3 if color == 2 else 4
        elif kind == b'IDAT':
            if chunks[-1] not in (b'IHDR', b'IDAT'):
                raise ValueError('nonconsecutive PNG IDAT')
            compressed.extend(body)
        elif kind == b'IEND':
            if size or not compressed or end != len(data):
                raise ValueError('invalid PNG IEND/trailing bytes')
        else:
            raise ValueError('unsupported PNG chunk')
        chunks.append(kind)
        offset = end
    if not chunks or chunks[-1] != b'IEND':
        raise ValueError('PNG missing IEND')
    expected = height * (1 + width * channels)
    inflater = zlib.decompressobj()
    raster = inflater.decompress(compressed, expected + 1)
    if (len(raster) != expected or not inflater.eof or inflater.unused_data or inflater.unconsumed_tail):
        raise ValueError('invalid PNG compressed raster length')
    if any(raster[row] > 4 for row in range(0, expected, 1 + width * channels)):
        raise ValueError('invalid PNG filter')
    return {'width': width, 'height': height, 'bytes': len(data)}


def load_manifest(path, max_bytes, max_pixels):
    raw = read_bounded(path, 1048576)
    document = parse_json(raw)
    if not isinstance(document, dict) or document.get('version') != 1:
        raise ValueError('manifest version must be 1')
    cases = document.get('cases')
    if not isinstance(cases, list) or not 1 <= len(cases) <= 64:
        raise ValueError('manifest requires 1..64 cases')
    seen, loaded = set(), []
    for case in cases:
        if not isinstance(case, dict):
            raise TypeError('case must be an object')
        case_id = case.get('id')
        if not isinstance(case_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', case_id) or case_id in seen:
            raise ValueError('invalid or duplicate case id')
        seen.add(case_id)
        if not isinstance(case.get('prompt'), str) or not 1 <= len(case['prompt']) <= 16384:
            raise ValueError('invalid prompt bounds')
        image_path = case.get('image')
        if (not isinstance(image_path, str) or not image_path or Path(image_path).is_absolute()
                or '..' in Path(image_path).parts):
            raise ValueError('image path must stay within manifest directory')
        digest = case.get('sha256')
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('sha256 must be exactly 64 lowercase hexadecimal characters')
        image = read_bounded(path.parent / image_path, max_bytes)
        if hashlib.sha256(image).hexdigest() != digest:
            raise ValueError('image sha256 mismatch')
        info = validate_png(image, max_pixels)
        loaded.append((case, image, info))
    return raw, loaded
