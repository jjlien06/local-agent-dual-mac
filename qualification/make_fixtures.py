"""Reproduce the public synthetic control fixtures into a NEW directory."""
import argparse
import hashlib
import json
import struct
import zlib

from qualification.store import Output


def make_png(mode):
    def chunk(kind, body):
        return struct.pack('!I', len(body)) + kind + body + struct.pack('!I', zlib.crc32(kind + body))
    red, blue = b'\xff\x00\x00', b'\x00\x00\xff'
    row = {'red': red * 128, 'blue': blue * 128, 'split': red * 64 + blue * 64}[mode]
    raster = (b'\x00' + row) * 128
    # One stored DEFLATE block avoids compressor-version-dependent fixture bytes.
    compressed = b'\x78\x01\x01' + struct.pack('<HH', len(raster), 65535 - len(raster)) + raster + struct.pack('!I', zlib.adler32(raster))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!IIBBBBB', 128, 128, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', compressed) + chunk(b'IEND', b''))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Output(args.output)
    try:
        cases = []
        for mode in ('red', 'blue', 'split'):
            image = make_png(mode)
            name = f'{mode}.png'
            output.write(name, image)
            prompt = ('Name the color filling this image. Reply with one lowercase word only.' if mode != 'split'
                      else 'Name the color on the LEFT half of this image. Reply with one lowercase word only.')
            cases.append({'id': mode, 'prompt': prompt, 'image': name,
                          'sha256': hashlib.sha256(image).hexdigest(),
                          'reference_output': 'blue' if mode == 'blue' else 'red'})
        document = {'version': 1, 'description': 'Public synthetic 128x128 RGB control; not a capability benchmark.', 'cases': cases}
        output.write('public-cases.json', (json.dumps(document, indent=2) + '\n').encode())
    finally:
        output.close()


if __name__ == '__main__':
    main()
