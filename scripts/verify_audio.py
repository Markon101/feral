#!/usr/bin/env python3
"""Inspect generated IEEE-float WAVs using only Python's standard library.

Does not modify recordings. Exact comparison is against the decoded canonical WAV.
"""
import argparse
import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


def read_float_wav(path):
    raw = Path(path).read_bytes()
    if raw[:4] != b'RIFF' or raw[8:12] != b'WAVE':
        raise ValueError('expected RIFF WAVE')
    offset, fmt, data = 12, None, None
    while offset + 8 <= len(raw):
        tag, size = struct.unpack_from('<4sI', raw, offset)
        payload = raw[offset + 8:offset + 8 + size]
        if tag == b'fmt ':
            fmt = list(struct.unpack_from('<HHIIHH', payload))
            if fmt[0] == 0xfffe and len(payload) >= 40:
                float_guid = bytes.fromhex('0300000000001000800000aa00389b71')
                if payload[24:40] == float_guid:
                    fmt[0] = 3
        if tag == b'data':
            data = payload
        offset += 8 + size + size % 2
    if fmt is None or data is None or fmt[0] != 3 or fmt[1] != 1 or fmt[5] != 32:
        raise ValueError('expected mono f32 WAV')
    x = array.array('f')
    x.frombytes(data)
    if sys.byteorder != 'little':
        x.byteswap()
    if not x or not all(math.isfinite(v) for v in x):
        raise ValueError('empty or nonfinite audio')
    return fmt[2], x, data


def stats(path):
    sr, x, _ = read_float_wav(path)
    return {'path': str(path), 'sample_rate': sr, 'samples': len(x),
            'seconds': len(x)/sr, 'peak': max(abs(v) for v in x),
            'rms': math.sqrt(sum(v*v for v in x)/len(x)),
            'samples_at_safety_limit': sum(abs(v) >= 0.979999 for v in x),
            'max_adjacent_difference': max(abs(a-b) for a, b in zip(x, x[1:])),
            'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio', nargs='+', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--continued', type=Path)
    parser.add_argument('--inpainted', type=Path)
    parser.add_argument('--start', type=float, default=1.)
    parser.add_argument('--end', type=float, default=1.25)
    args = parser.parse_args()
    result = {'audio': [stats(p) for p in args.audio]}
    if args.reference:
        sr, ref, raw = read_float_wav(args.reference)
        if args.continued:
            rate, x, data = read_float_wav(args.continued)
            exact = rate == sr and data[:len(raw)] == raw
            assert exact and len(x) > len(ref), 'continuation changed canonical prefix'
            i = len(ref)
            result['continuation'] = {'exact_prefix': exact,
                                      'new_samples': len(x)-i,
                                      'boundary_jump': abs(x[i]-x[i-1]),
                                      'boundary_derivative_change': abs((x[i]-x[i-1])-(x[i-1]-x[i-2]))}
        if args.inpainted:
            rate, x, data = read_float_wav(args.inpainted)
            a, b = round(args.start*sr), round(args.end*sr)
            exact = rate == sr and len(data) == len(raw) and data[:a*4] == raw[:a*4] and data[b*4:] == raw[b*4:]
            assert exact and data[a*4:b*4] != raw[a*4:b*4], 'inpainting preservation/change failed'
            result['inpainting'] = {'exact_outside': exact, 'changed_interval': True,
                                    'boundary_jumps': [abs(x[a]-x[a-1]), abs(x[b]-x[b-1])]}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
