#!/usr/bin/env python3
"""Create new constant-gain crops from an existing prepared FERAL dataset.

The parent directory and original recordings remain unchanged. No clipping,
compression, EQ, peak normalization or source selection occurs here.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import struct

from prepare_scaling_data import RATE, sha256, verify_wav


def float_payload(path):
    raw = Path(path).read_bytes()
    offset = 12
    while offset + 8 <= len(raw):
        tag, size = struct.unpack_from('<4sI', raw, offset)
        if tag == b'data':
            return raw[offset + 8:offset + 8 + size]
        offset += 8 + size + size % 2
    raise ValueError('WAV has no PCM data')


def write_gained(path, payload, gain):
    data = bytearray(len(payload))
    for i, (sample,) in enumerate(struct.iter_unpack('<f', payload)):
        gained = sample * gain
        if not math.isfinite(gained):
            raise ValueError('constant gain produced nonfinite audio')
        struct.pack_into('<f', data, i * 4, gained)
    with Path(path).open('xb') as f:
        f.write(struct.pack('<4sI4s4sIHHIIHH4sI', b'RIFF', 36 + len(data), b'WAVE',
                            b'fmt ', 16, 3, 1, RATE, RATE * 4, 4, 32, b'data', len(data)))
        f.write(data)


def level_match_prepared(prepared_dir, output_dir, target_rms):
    if not math.isfinite(target_rms) or not 0 < target_rms <= 1:
        raise ValueError('target RMS must be finite and in (0,1]')
    parent = Path(prepared_dir).resolve(strict=True)
    output = Path(output_dir).absolute()
    if output.exists():
        raise FileExistsError(f'refusing existing output directory: {output}')
    parent_file = parent / 'provenance.json'
    parent_hash = sha256(parent_file)
    original = json.loads(parent_file.read_text())
    if (original.get('sample_rate'), original.get('channels'), original.get('encoding')) != (RATE, 1, 'pcm_f32le'):
        raise ValueError('parent must be prepared mono 16k f32 data')
    crops = original['crops']
    if not crops or len(crops) > 128:
        raise ValueError('parent must contain 1..128 crops')
    if sum(c['samples'] for c in crops) > 32_000_000:
        raise ValueError('parent exceeds the decoded 128 MB bound')
    output.mkdir(parents=True, exist_ok=False)
    metadata = {**original, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'level_matching': {'kind': 'per-crop-constant-rms-gain', 'target_rms': target_rms,
                                   'parent_provenance': str(parent_file),
                                   'parent_provenance_sha256': parent_hash,
                                   'clipping': False, 'equalization': False}, 'crops': []}
    with (output / 'plan.json').open('x') as f:
        json.dump(metadata, f, indent=2, allow_nan=False)
        f.write('\n')
    manifests = {name: [] for name in ('train', 'validation', 'test')}
    for crop in crops:
        name = crop['path']
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('parent crop paths must be single relative filenames')
        if crop['split'] not in manifests:
            raise ValueError('unknown parent split')
        source = parent / name
        raw_stats = verify_wav(source, crop['samples'])
        if raw_stats['sha256'] != crop['sha256']:
            raise ValueError(f'parent crop hash mismatch: {source}')
        if raw_stats['rms'] <= 0:
            raise ValueError(f'cannot RMS-match silent crop: {source}')
        gain = target_rms / raw_stats['rms']
        payload = float_payload(source)
        destination_name = 'matched-' + name
        destination = output / destination_name
        write_gained(destination, payload, gain)
        stats = verify_wav(destination, crop['samples'])
        if abs(stats['rms'] - target_rms) > 1e-7 * target_rms:
            raise ValueError('float output RMS differs from target beyond rounding tolerance')
        if sha256(source) != raw_stats['sha256']:
            raise ValueError('parent crop changed during matching')
        metadata['crops'].append({**crop, **stats, 'path': destination_name,
                                  'raw_crop': {'path': str(source), **raw_stats},
                                  'gain': gain, 'target_rms': target_rms})
        manifests[crop['split']].append(destination_name)
    if sha256(parent_file) != parent_hash:
        raise ValueError('parent provenance changed during matching')
    for split, names in manifests.items():
        filename = 'manifest.txt' if split == 'train' else f'{split}.txt'
        with (output / filename).open('x') as f:
            f.write('# FERAL constant-gain crops; use --no-normalize\n' + '\n'.join(names) + '\n')
        metadata[split + '_manifest_sha256'] = sha256(output / filename)
    with (output / 'provenance.json').open('x') as f:
        json.dump(metadata, f, indent=2, allow_nan=False)
        f.write('\n')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='existing prepared directory')
    parser.add_argument('--output', type=Path, required=True, help='new directory only')
    parser.add_argument('--target-rms', type=float, required=True)
    args = parser.parse_args()
    try:
        result = level_match_prepared(args.input, args.output, args.target_rms)
    except (ValueError, OSError, KeyError, OverflowError) as exc:
        parser.exit(1, f'FERAL level matching failed: {exc}\n')
    print(json.dumps({'output': str(args.output.absolute()), 'crops': len(result['crops']),
                      'target_rms': args.target_rms,
                      'parent_provenance_sha256': result['level_matching']['parent_provenance_sha256']}))


if __name__ == '__main__':
    main()
