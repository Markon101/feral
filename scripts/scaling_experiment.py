#!/usr/bin/env python3
"""Sequential fixed 2x2 data/capacity sweep; resume only its owned progress ledger.

No automatic fallback or checkpoint overwrite. Same model/objective/crop/LR except
for the declared width and data axes. See reports/SCALING_PROTOCOL.md.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from device_budget import snapshot


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_progress(root, state):
    temp=root/f'.progress-{time.monotonic_ns()}.tmp'
    with temp.open('x') as f:
        json.dump(state,f,indent=2,allow_nan=False)
        f.write('\n')
    temp.replace(root/'progress.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--expanded',type=Path,required=True)
    p.add_argument('--baseline',type=Path,default=Path('runs/phase2-v1/train.wav'))
    p.add_argument('--steps',type=int,default=20000)
    p.add_argument('--resume-sweep',action='store_true')
    p.add_argument('--only-width',type=int,choices=[64,128],help='run a subset without claiming the full sweep completed')
    a=p.parse_args()
    if not 10000<=a.steps<=100000 or a.steps%2000:
        p.error('--steps must be a multiple of2000 in10000..100000')
    root=a.run.resolve();binary=Path('target/release/feral').resolve(strict=True)
    launcher=Path('scripts/feral-opencl.sh').resolve(strict=True)
    inputs={'old24':a.baseline.resolve(strict=True),'expanded200':a.expanded.resolve(strict=True)}
    input_hashes={k:sha(v) for k,v in inputs.items()}
    if a.resume_sweep:
        state=json.loads((root/'progress.json').read_text())
        plan=state['plan']
        if plan['binary_sha256']!=sha(binary) or plan['input_hashes']!=input_hashes or plan['steps']!=a.steps:
            raise ValueError('resume requires identical binary, input identities and target')
    else:
        root.mkdir(parents=True,exist_ok=False)
        plan={'started_utc':utc(),'steps':a.steps,'seeds':[42,43],
              'widths':[64,128],'input_hashes':input_hashes,'inputs':{k:str(v) for k,v in inputs.items()},
              'binary_sha256':sha(binary),'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'baseline':'all fresh; no100k weights reused','fixed':'pyramid8192samples16kmono,scale/prior.1,lr.0003,uniform-time residual MSE',
              'monitor':'read-only cpuset/affinity/thermal/memory eachsecond; trainer RSS limit2GiB; no hardware setting changes'}
        (root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        state={'plan':plan,'arms':{}}
        write_progress(root,state)
    def event(value):
        with (root/'events.jsonl').open('a') as f:
            f.write(json.dumps({'utc':utc(),**value},allow_nan=False)+'\n')
    for seed in [42,43]:
        axes=[(64,'old24'),(64,'expanded200'),(128,'old24'),(128,'expanded200')]
        if seed==43:
            axes.reverse()
        for width,data in axes:
            if a.only_width is not None and width != a.only_width:
                continue
            key=f'w{width}-{data}-s{seed}'
            arm=state['arms'].setdefault(key,{'width':width,'data':data,'seed':seed,'completed':0,'chunks':[],'milestones':{}})
            while arm['completed']<a.steps:
                budget=snapshot()
                if not budget['allow_training']:
                    event({'event':'resource_deferred','arm':key,'budget':budget})
                    write_progress(root,state)
                    print('RESOURCE_DEFERRED',json.dumps(budget),flush=True)
                    return 75
                if sha(binary)!=plan['binary_sha256'] or any(sha(v)!=input_hashes[k] for k,v in inputs.items()):
                    raise ValueError('binary or input changed during experiment')
                steps=min(budget['chunk_steps'],a.steps-arm['completed'])
                # Stop on10k milestones even if scheduling changes.
                steps=min(steps,10000-arm['completed']%10000)
                parent=root/key;parent.mkdir(exist_ok=True)
                index=len(list(parent.glob('chunk-*')))
                folder=parent/f'chunk-{index:03d}'
                argv=[str(launcher),'train','--backend','opencl','--network','pyramid','--width',str(width),
                      '--input',str(inputs[data]),'--run',str(folder),'--steps',str(steps),'--seed',str(seed),
                      '--parameterization','preconditioned','--data-scale','.1','--prior-std','.1',
                      '--no-normalize','--preview-every',str(steps),'--checkpoint-every',str(min(1000,steps))]
                if arm['completed']:
                    argv+=['--resume',arm['checkpoint']]
                event({'event':'start_chunk','arm':key,'completed':arm['completed'],'argv':argv,'budget':budget})
                now=time.monotonic();peak_rss=0;observations=[]
                with (parent/f'chunk-{index:03d}.log').open('x') as log:
                    child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    while child.poll() is None:
                        observation=snapshot()
                        try:
                            for line in Path(f'/proc/{child.pid}/status').read_text().splitlines():
                                if line.startswith(('VmHWM:','VmRSS:')):
                                    peak_rss=max(peak_rss,int(line.split()[1]))
                        except OSError:
                            pass
                        observations.append({'elapsed_seconds':time.monotonic()-now,'budget':observation,'host_rss_high_kib':peak_rss})
                        if not observation['allow_training'] or peak_rss>=2*1024*1024:
                            os.killpg(child.pid,signal.SIGTERM)
                            child.wait(timeout=15)
                            event({'event':'interrupted_resource_limit','arm':key,'observations':observations,'folder':str(folder)})
                            write_progress(root,state)
                            print('RESOURCE_INTERRUPTED checkpoint files preserved',folder,flush=True)
                            return 75
                        try:
                            child.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            pass
                    if child.returncode:
                        event({'event':'child_failed','arm':key,'returncode':child.returncode,'folder':str(folder),'observations':observations})
                        raise RuntimeError(f'trainer exited{child.returncode}; see{folder}')
                checkpoint=folder/'checkpoint.json'
                ck=json.loads(checkpoint.read_text());metrics=json.loads((folder/'metrics.json').read_text())
                expected=arm['completed']+steps
                if ck['completed_steps']!=expected or ck['adam']['step']!=expected or metrics['backend']!='opencl-f32-resident':
                    raise ValueError('checkpoint/backend contract failed')
                arm['completed']=expected;arm['checkpoint']=str(checkpoint)
                if expected%10000==0:
                    arm['milestones'][str(expected)]=str(checkpoint)
                record={'additional_steps':steps,'total_steps':expected,'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
                        'wall_seconds':time.monotonic()-now,'metrics':metrics,'observations':observations}
                arm['chunks'].append(record)
                event({'event':'end_chunk','arm':key,**record})
                write_progress(root,state)
                print(f"{key} {expected}/{a.steps} | step{metrics['step_mean_ms']:.3f}ms | probe{metrics['fixed_probe_final_loss']:.5f} | {budget['scheduling_state']}",flush=True)
    expected=[f'w{w}-{d}-s{s}' for s in [42,43] for w in [64,128] for d in ['old24','expanded200']]
    complete=all(state['arms'].get(k,{}).get('completed',0)==a.steps for k in expected)
    state['finished_utc' if complete else 'last_partial_utc']=utc();write_progress(root,state)
    print('COMPLETE' if complete else 'PARTIAL_SELECTION',root/'progress.json',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
