#!/usr/bin/env python3
"""Prepare bounded, provenance-recorded audio splits without modifying sources.

Windows are source seconds, half open START:END. Every training window must avoid
all evaluation/exclusion windows plus --guard-seconds. Historical AROUND evaluation
windows are protected by default. Outputs are unnormalized 16 kHz mono float WAVs.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
from datetime import datetime, timezone

RATE = 16000
HISTORICAL = ('112:120', '136:144', '160:168', '184:192', '216:224', '232:240')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def window(text):
    try:
        start, end = map(float, text.split(':'))
    except (ValueError, TypeError):
        raise ValueError('window must be START:END seconds') from None
    if not all(math.isfinite(v) for v in (start, end)) or start < 0 or end <= start:
        raise ValueError('window must have finite 0 <= START < END')
    a, b = round(start * RATE), round(end * RATE)
    if b <= a:
        raise ValueError('window must contain at least one sample')
    return a, b


def overlaps(a, b):
    return max(a[0], b[0]) < min(a[1], b[1])


def plan_windows(train, validation, test, exclusions, duration, guard=2., chunk=20.):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('source duration must be finite and positive')
    if not math.isfinite(guard) or not 0 <= guard <= 60:
        raise ValueError('guard seconds must be in 0..60')
    if not math.isfinite(chunk) or not 0 < chunk <= 30:
        raise ValueError('chunk seconds must be in (0,30] to avoid loader truncation')
    groups = {'train': list(train), 'validation': list(validation), 'test': list(test)}
    if not train or not validation or not test:
        raise ValueError('explicit train, validation and test windows are required')
    limit = round(duration * RATE)
    all_windows = [(name, item) for name, items in groups.items() for item in items]
    for i, (name, item) in enumerate(all_windows):
        if item[0] < 0 or item[1] > limit or item[0] >= item[1]:
            raise ValueError(f'{name} window {item} lies outside source duration')
        for other_name, other in all_windows[i + 1:]:
            if overlaps(item, other):
                raise ValueError(f'overlapping {name}/{other_name} source windows')
    protected = list(validation) + list(test) + list(exclusions)
    margin = round(guard * RATE)
    for item in train:
        if any(overlaps(item, (max(0, p[0] - margin), p[1] + margin)) for p in protected):
            raise ValueError('training window overlaps a protected evaluation window or guard')
    size = max(1, round(chunk * RATE))
    result = []
    for split, items in groups.items():
        for a, b in items:
            for start in range(a, b, size):
                result.append({'split': split, 'start_sample': start,
                               'end_sample': min(start + size, b)})
    if len(result) > 128:
        raise ValueError('more than 128 prepared files exceeds the training loader limit')
    if sum(r['end_sample'] - r['start_sample'] for r in result) * 4 > 128_000_000:
        raise ValueError('prepared PCM exceeds the 128 MB decoded dataset bound')
    return result


def decoder_env():
    env = os.environ.copy()
    for name in ('LD_LIBRARY_PATH', 'LD_PRELOAD'):
        saved = env.get('FERAL_DECODER_' + name)
        if saved is not None:
            if saved:
                env[name] = saved
            else:
                env.pop(name, None)
    cpp = Path('/data/data/com.termux/files/usr/lib/libc++_shared.so')
    if cpp.exists():
        old = env.get('LD_PRELOAD', '')
        env['LD_PRELOAD'] = str(cpp) + (':' + old if old else '')
    return env


def run(argv):
    result = subprocess.run(argv, env=decoder_env(), capture_output=True, text=True,
                            timeout=120, check=False)
    if result.returncode:
        raise ValueError(f'{argv[0]} failed ({result.returncode}): {result.stderr[-4000:]}')
    return result.stdout


def verify_wav(path, expected_samples):
    # At most 30 s per crop: bounded <2 MB WAV read, no whole-song PCM allocation.
    raw = Path(path).read_bytes()
    if raw[:4] != b'RIFF' or raw[8:12] != b'WAVE':
        raise ValueError('decoder did not produce RIFF WAVE')
    offset, fmt, data = 12, None, None
    while offset + 8 <= len(raw):
        tag, size = struct.unpack_from('<4sI', raw, offset)
        payload = raw[offset + 8:offset + 8 + size]
        if len(payload) != size:
            raise ValueError('truncated WAV chunk')
        if tag == b'fmt ':
            fmt = list(struct.unpack_from('<HHIIHH', payload))
            if fmt[0] == 0xfffe and payload[24:40] == bytes.fromhex('0300000000001000800000aa00389b71'):
                fmt[0] = 3
        if tag == b'data':
            data = payload
        offset += 8 + size + size % 2
    if fmt is None or fmt[0:3] != [3, 1, RATE] or fmt[5] != 32 or data is None:
        raise ValueError('decoder did not produce canonical mono 16k f32 PCM')
    if len(data) != expected_samples * 4:
        raise ValueError(f'decoded sample count {len(data)//4} != {expected_samples}')
    energy, peak = 0., 0.
    for (value,) in struct.iter_unpack('<f', data):
        if not math.isfinite(value):
            raise ValueError('nonfinite decoded sample')
        energy += value * value
        peak = max(peak, abs(value))
    return {'samples': expected_samples, 'rms': math.sqrt(energy / expected_samples),
            'peak': peak, 'sha256': sha256(path)}


def prepare(source, output, train, validation, test, exclusions, guard=2., chunk=20.):
    source = Path(source).resolve(strict=True)
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError(f'refusing existing output directory: {output}')
    if '\n' in str(source) or '\r' in str(source):
        raise ValueError('source paths with newlines cannot be represented in manifests')
    probe_argv = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                  '-of', 'json', str(source)]
    duration = float(json.loads(run(probe_argv))['format']['duration'])
    plan = plan_windows(train, validation, test, exclusions, duration, guard, chunk)
    before = sha256(source)
    output.mkdir(parents=True, exist_ok=False)
    metadata = {'version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'source': str(source), 'source_sha256': before, 'source_duration': duration,
                'probe_argv': probe_argv, 'sample_rate': RATE, 'channels': 1,
                'encoding': 'pcm_f32le', 'normalize': False, 'guard_seconds': guard,
                'requested_windows_samples': {'train': train, 'validation': validation, 'test': test},
                'excluded_windows_samples': exclusions, 'chunk_seconds': chunk, 'crops': []}
    # Preserve a plan before decoding; a failed attempt remains inspectable.
    (output / 'plan.json').write_text(json.dumps(metadata, indent=2) + '\n')
    manifests = {name: [] for name in ('train', 'validation', 'test')}
    for i, crop in enumerate(plan):
        name = f"{crop['split']}-{i:03d}.wav"
        destination = output / name
        start, end = crop['start_sample'], crop['end_sample']
        # Resample then sample-index trim. No input seek/keyframe approximation;
        # filter context comes from the original stream and output boundaries are exact.
        audio_filter = f'aresample={RATE},atrim=start_sample={start}:end_sample={end},asetpts=PTS-STARTPTS'
        argv = ['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1', '-i', str(source),
                '-map', '0:a:0', '-vn', '-sn', '-dn', '-af', audio_filter,
                '-ac', '1', '-ar', str(RATE), '-c:a', 'pcm_f32le', '-n', str(destination)]
        run(argv)
        signal = verify_wav(destination, end - start)
        metadata['crops'].append({**crop, 'path': name, 'offset_seconds': start / RATE,
                                  'duration_seconds': (end - start) / RATE,
                                  'decode_argv': argv, **signal})
        manifests[crop['split']].append(name)
    if sha256(source) != before:
        raise ValueError('source changed during preparation; manifests were not published')
    metadata['source_sha256_after'] = before
    for split, names in manifests.items():
        filename = 'manifest.txt' if split == 'train' else f'{split}.txt'
        text = '# FERAL canonical crops; no gain normalization\n' + '\n'.join(names) + '\n'
        with (output / filename).open('x') as f:
            f.write(text)
        metadata[split + '_manifest_sha256'] = sha256(output / filename)
    with (output / 'provenance.json').open('x') as f:
        json.dump(metadata, f, indent=2, allow_nan=False)
        f.write('\n')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    for split in ('train', 'validation', 'test'):
        parser.add_argument('--' + split + '-window', action='append', type=window, required=True)
    parser.add_argument('--exclude-window', action='append', type=window, default=[])
    parser.add_argument('--guard-seconds', type=float, default=2.)
    parser.add_argument('--chunk-seconds', type=float, default=20.)
    args = parser.parse_args()
    exclusions = [window(s) for s in HISTORICAL] + args.exclude_window
    try:
        result = prepare(args.source, args.output, args.train_window, args.validation_window,
                         args.test_window, exclusions, args.guard_seconds, args.chunk_seconds)
    except (OSError, ValueError, subprocess.TimeoutExpired, KeyError) as exc:
        parser.exit(1, f'FERAL preparation failed: {exc}\n')
    print(json.dumps({'output': str(args.output.absolute()), 'crops': len(result['crops']),
                      'source_sha256': result['source_sha256']}))


if __name__ == '__main__':
    main()
