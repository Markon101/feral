#!/usr/bin/env python3
"""Fixed follow-up after phase2.py: twice the pooled updates, fresh held-out sections.

This is an approximate compute-budget comparison, NOT an equal-update ablation.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
from phase2 import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', required=True, type=Path)
    p.add_argument('--run', required=True, type=Path)
    a = p.parse_args()
    base, root = a.base.resolve(strict=True), a.run.resolve()
    prior = json.loads((base/'plan.json').read_text())
    source = Path(prior['source'])
    assert sha(source) == prior['source_sha256']
    root.mkdir(parents=True, exist_ok=False)
    binary = Path('target/release/feral').resolve(strict=True)
    steps = prior['steps_per_model']
    plan = {'protocol': 'phase2-budget-v1', 'base': str(base), 'base_plan_sha256': sha(base/'plan.json'),
            'additional_pyramid_steps': steps, 'baseline_total_steps': steps,
            'pyramid_total_steps': steps*2, 'seeds': prior['seeds'],
            'new_split_seconds': {'validation': [64,72], 'test': [80,88]},
            'source_sha256': sha(source), 'binary_sha256': sha(binary), 'script_sha256': sha(__file__),
            'commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
            'reason': 'equal-step pyramid was worse; test whether cheaper updates improve the practical compute tradeoff',
            'limits': 'new held-out sections; baseline saw fewer updates/crops; not evidence that long context caused any advantage'}
    (root/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    env = os.environ.copy()
    cpp = Path('/data/data/com.termux/files/usr/lib/libc++_shared.so')
    if cpp.exists():
        env['LD_PRELOAD'] = str(cpp) + (':'+env['LD_PRELOAD'] if env.get('LD_PRELOAD') else '')
    count = 0

    def run(argv):
        nonlocal count
        count += 1
        argv = [str(x) for x in argv]
        print(f'[{count}] '+' '.join(argv), flush=True)
        with (root/'commands.jsonl').open('a') as f:
            f.write(json.dumps({'index': count, 'argv': argv})+'\n')
        with (root/f'command-{count:02d}.log').open('x') as log:
            subprocess.run(argv, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)

    for name, (start,end) in plan['new_split_seconds'].items():
        run(['ffmpeg','-nostdin','-v','error','-n','-ss',start,'-threads','1','-i',source,'-t',end-start,'-map','0:a:0','-ac','1','-ar','16000','-c:a','pcm_f32le',root/f'{name}.wav'])
    cases = []
    for seed in prior['seeds']:
        case = root/f'pyramid-{seed}'
        run([binary,'train','--input',base/'train.wav','--run',case,'--network','pyramid',
             '--steps',steps,'--seed',seed,'--no-normalize','--preview-every',steps+1,
             '--checkpoint-every',steps+1,'--resume',base/f'pyramid-{seed}'/'checkpoint.json'])
        for name, checkpoint in [('baseline600',base/f'baseline-{seed}'/'checkpoint.json'),
                                  ('pyramid600',base/f'pyramid-{seed}'/'checkpoint.json'),
                                  ('pyramid1200',case/'checkpoint.json')]:
            evaluations = {}
            for split in ['validation','test']:
                out = root/f'{name}-{seed}-{split}.json'
                run([binary,'evaluate','--checkpoint',checkpoint,'--input',root/f'{split}.wav',
                     '--output',out,'--seed','7001','--crops','4'])
                evaluations[split] = json.loads(out.read_text())
            cases.append({'name':name,'seed':seed,'evaluations':evaluations})
        run([binary,'analyze','--input',case/'generated.wav','--reference',base/'train.wav','--output',case/'signal.json'])
    assert sha(source) == prior['source_sha256']
    (root/'results.json').write_text(json.dumps({'plan':plan,'cases':cases,'source_unchanged':True},indent=2)+'\n')
    print('Completed follow-up: '+str(root/'results.json'),flush=True)


if __name__ == '__main__':
    main()
