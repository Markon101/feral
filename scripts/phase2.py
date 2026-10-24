#!/usr/bin/env python3
"""Fixed small CPU comparison: baseline vs pooled pyramid, three model seeds.

Run from the FERAL root. Local source is read only; output root must be new.
No packages, models, services or Python dependencies are downloaded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(65536), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--steps', type=int, default=600)
    a = p.parse_args()
    if not 1 <= a.steps <= 2000:
        p.error('steps must be 1..2000 for this bounded protocol')
    root = a.run.resolve()
    source = a.input.resolve(strict=True)
    binary = Path('target/release/feral').resolve(strict=True)
    root.mkdir(parents=True, exist_ok=False)
    source_hash = sha(source)
    plan = {'protocol': 'phase2-pyramid-v1', 'source': str(source),
            'source_sha256': source_hash, 'binary_sha256': sha(binary),
            'script_sha256': sha(__file__),
            'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'seeds': [42, 43, 44], 'steps_per_model': a.steps,
            'split_seconds': {'train': [0, 24], 'validation': [32, 40], 'test': [48, 56]},
            'normalization': 'none, same raw gain for train and held-out audio',
            'crop_seconds': 0.512, 'learning_rate': 0.0003,
            'evaluation_seed': 7001, 'evaluation_crops': 4,
            'selection': 'fixed plan before training, no checkpoint/seed selection',
            'limits': 'one recording; no cross-recording generalization or listening claim'}
    (root/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    counter = 0
    env = os.environ.copy()
    cpp = Path('/data/data/com.termux/files/usr/lib/libc++_shared.so')
    if cpp.exists():
        env['LD_PRELOAD'] = str(cpp) + (':'+env['LD_PRELOAD'] if env.get('LD_PRELOAD') else '')

    def run(args):
        nonlocal counter
        counter += 1
        args = [str(x) for x in args]
        print(f'[{counter}] '+ ' '.join(args), flush=True)
        with (root/'commands.jsonl').open('a') as out:
            out.write(json.dumps({'index': counter, 'argv': args})+'\n')
        with (root/f'command-{counter:02d}.log').open('x') as log:
            subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True, env=env)

    for name, (start, end) in plan['split_seconds'].items():
        run(['ffmpeg', '-nostdin', '-v', 'error', '-n', '-ss', start,
             '-threads', '1', '-i', source, '-t', end-start, '-map', '0:a:0',
             '-ac', '1', '-ar', '16000', '-c:a', 'pcm_f32le', root/f'{name}.wav'])
    results = []
    for seed in plan['seeds']:
        networks = ['baseline', 'pyramid'] if seed % 2 == 0 else ['pyramid', 'baseline']
        for network in networks:
            case = root/f'{network}-{seed}'
            run([binary, 'train', '--input', root/'train.wav', '--run', case,
                 '--network', network, '--steps', a.steps, '--seed', seed,
                 '--no-normalize', '--preview-every', a.steps+1,
                 '--checkpoint-every', a.steps+1])
            for split in ['train', 'validation', 'test']:
                run([binary, 'evaluate', '--checkpoint', case/'checkpoint.json',
                     '--input', root/f'{split}.wav', '--output', case/f'eval-{split}.json',
                     '--crops', '4', '--seed', '7001'])
            run([binary, 'analyze', '--input', case/'generated.wav',
                 '--reference', root/'train.wav', '--output', case/'signal.json'])
            results.append({'network': network, 'seed': seed,
                            'metrics': json.loads((case/'metrics.json').read_text()),
                            'signal': json.loads((case/'signal.json').read_text()),
                            'evaluations': {s: json.loads((case/f'eval-{s}.json').read_text()) for s in ['train','validation','test']}})
    # Assert batch/mask/noise RNG traces really match within every architecture pair.
    for seed in plan['seeds']:
        traces = []
        for network in ['baseline', 'pyramid']:
            rows = [json.loads(l) for l in (root/f'{network}-{seed}'/'losses.jsonl').read_text().splitlines()]
            traces.append([(r['task'], r['t'], r['rng_state']) for r in rows])
        assert traces[0] == traces[1], 'paired training streams diverged'
    assert source_hash == sha(source), 'source recording changed during protocol'
    (root/'results.json').write_text(json.dumps({'plan': plan, 'paired_rng_traces_equal': True,
                                               'source_unchanged': True, 'cases': results}, indent=2)+'\n')
    print('Completed fixed paired protocol: '+str(root/'results.json'), flush=True)


if __name__ == '__main__':
    main()
