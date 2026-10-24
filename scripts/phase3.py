#!/usr/bin/env python3
"""Fixed sequential raw-velocity vs preconditioned-flow comparison, three model seeds."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from phase2 import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--steps',type=int,default=3000)
    args=p.parse_args()
    if not 1<=args.steps<=6000: p.error('steps must be 1..6000')
    root=args.run.resolve();root.mkdir(parents=True,exist_ok=False)
    source=Path('/storage/emulated/0/Download/AROUND (The Absence of Ladder)v2.wav')
    train=Path('runs/phase2-v1/train.wav').resolve(strict=True)
    binary=Path('target/release/feral').resolve(strict=True)
    controls=Path('target/release/examples/phase3_controls').resolve(strict=True)
    plan={'protocol':'phase3-precondition-v1','steps_per_model':args.steps,'seeds':[42,43,44],
          'network':'pyramid','parameters':57952,'data_scale':0.1,'prior_noise_std':1,
          'train':str(train),'train_sha256':sha(train),'train_source_seconds':[0,24],
          'new_splits':{'validation':[112,120],'test':[136,144]},
          'source':str(source),'source_sha256':sha(source),'binary_sha256':sha(binary),
          'script_sha256':sha(__file__),'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'sampler':{'solver':'heun','steps':32,'seed':42},
          'selection':'fixed final checkpoints, no seed selection; seed42 listening examples predetermined',
          'limits':'same-song sections; preconditioning changes input/output scaling and time loss weights together; no claim that regression improvement equals perception'}
    (root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    env=os.environ.copy();cpp=Path('/data/data/com.termux/files/usr/lib/libc++_shared.so')
    if cpp.exists():env['LD_PRELOAD']=str(cpp)+(':'+env['LD_PRELOAD'] if env.get('LD_PRELOAD') else '')
    count=0
    def run(argv):
        nonlocal count
        count+=1;argv=[str(x) for x in argv]
        print(f'[{count}] '+' '.join(argv),flush=True)
        with (root/'commands.jsonl').open('a') as f:f.write(json.dumps({'index':count,'argv':argv})+'\n')
        with (root/f'command-{count:02d}.log').open('x') as f:subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
    for name,(a,b) in plan['new_splits'].items():
        run(['ffmpeg','-nostdin','-v','error','-n','-ss',a,'-threads','1','-i',source,'-t',b-a,'-map','0:a:0','-ac','1','-ar','16000','-c:a','pcm_f32le',root/f'{name}.wav'])
    run([controls,'--reference',train,'--output',root/'controls'])
    cases=[]
    for seed in plan['seeds']:
        kinds=['velocity','preconditioned'] if seed%2==0 else ['preconditioned','velocity']
        for kind in kinds:
            case=root/f'{kind}-{seed}'
            run([binary,'train','--input',train,'--run',case,'--network','pyramid','--parameterization',kind,'--data-scale','0.1','--steps',args.steps,'--seed',seed,'--no-normalize','--preview-every',args.steps+1,'--checkpoint-every',max(1,args.steps//2)])
            for split in ['validation','test']:
                run([binary,'evaluate','--checkpoint',case/'checkpoint.json','--input',root/f'{split}.wav','--output',case/f'eval-{split}.json','--crops','4','--seed','7001'])
            run([binary,'generate','--checkpoint',case/'checkpoint.json','--seconds','0.512','--seed','42','--flow-steps','32','--output',case/'seed42-heun32.wav'])
            run([binary,'analyze','--input',case/'seed42-heun32.wav','--reference',train,'--output',case/'signal.json'])
            cases.append({'kind':kind,'seed':seed,'metrics':json.loads((case/'metrics.json').read_text()),'signal':json.loads((case/'signal.json').read_text()),'evaluations':{sp:json.loads((case/f'eval-{sp}.json').read_text()) for sp in ['validation','test']}})
    for seed in plan['seeds']:
        traces=[]
        for kind in ['velocity','preconditioned']:
            rows=[json.loads(l) for l in (root/f'{kind}-{seed}'/'losses.jsonl').read_text().splitlines()]
            traces.append([(r['step'],r['task'],r['t'],r['rng_state']) for r in rows])
        assert traces[0]==traces[1], 'paired crops/noise/masks diverged'
    assert sha(source)==plan['source_sha256'] and sha(train)==plan['train_sha256']
    (root/'results.json').write_text(json.dumps({'plan':plan,'cases':cases,'paired_rng_traces_equal':True,'sources_unchanged':True},indent=2)+'\n')
    print('COMPLETE '+str(root/'results.json'),flush=True)


if __name__=='__main__':main()
