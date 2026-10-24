#!/usr/bin/env python3
"""Sequential calibrated-prior follow-up with common-corruption endpoint evaluation."""
import argparse,json,os,subprocess
from pathlib import Path
from phase2 import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path);a=p.parse_args()
    root=a.run.resolve();root.mkdir(parents=True,exist_ok=False)
    train=Path('runs/phase2-v1/train.wav').resolve(strict=True)
    source=Path('/storage/emulated/0/Download/AROUND (The Absence of Ladder)v2.wav')
    binary=Path('target/release/feral').resolve(strict=True);snr=Path('target/release/examples/phase3_snr_eval').resolve(strict=True)
    plan={'protocol':'phase3-calibrated-v1','seeds':[42,43,44],'steps':3000,'data_scale':0.1,'prior_std':0.1,
          'network':'pyramid','train':str(train),'train_sha256':sha(train),'source':str(source),'source_sha256':sha(source),
          'new_splits':{'validation':[216,224],'test':[232,240]},'binary_sha256':sha(binary),'script_sha256':sha(__file__),
          'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'evaluation':'common y+sigma*epsilon, sigma=.02,.1,.5; t=prior_std/(prior_std+sigma); compare physical endpoint MSE, not velocity losses across priors',
          'selection':'fixed final3000, seed42 listening example predetermined; primary and unit6000 negative/positive outcomes retained'}
    (root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    env=os.environ.copy();cpp=Path('/data/data/com.termux/files/usr/lib/libc++_shared.so')
    if cpp.exists():env['LD_PRELOAD']=str(cpp)+(':'+env['LD_PRELOAD'] if env.get('LD_PRELOAD') else '')
    counter=0
    def run(argv):
        nonlocal counter
        counter+=1;argv=list(map(str,argv));print(f'[{counter}] '+' '.join(argv),flush=True)
        with (root/'commands.jsonl').open('a') as f:f.write(json.dumps({'index':counter,'argv':argv})+'\n')
        with (root/f'command-{counter:02d}.log').open('x') as f:subprocess.run(argv,check=True,stdout=f,stderr=subprocess.STDOUT,env=env)
    for name,(start,end) in plan['new_splits'].items():
        run(['ffmpeg','-nostdin','-v','error','-n','-ss',start,'-threads','1','-i',source,'-t',end-start,'-map','0:a:0','-ac','1','-ar','16000','-c:a','pcm_f32le',root/f'{name}.wav'])
    cases=[]
    for seed in plan['seeds']:
        case=root/f'calibrated-{seed}'
        run([binary,'train','--input',train,'--run',case,'--network','pyramid','--parameterization','preconditioned','--data-scale','0.1','--prior-std','0.1','--steps','3000','--seed',seed,'--no-normalize','--preview-every','3001','--checkpoint-every','1500'])
        run([binary,'generate','--checkpoint',case/'checkpoint.json','--seconds','0.512','--seed','42','--flow-steps','32','--output',case/'seed42-heun32.wav'])
        run([binary,'analyze','--input',case/'seed42-heun32.wav','--reference',train,'--output',case/'signal.json'])
        scores={}
        for name,ck in [('calibrated',case/'checkpoint.json'),('unit',Path(f'runs/phase3-v1/preconditioned-{seed}/checkpoint.json')),('velocity',Path(f'runs/phase3-v1/velocity-{seed}/checkpoint.json'))]:
            scores[name]={}
            for split in ['validation','test']:
                out=root/f'{name}-{seed}-{split}.json';run([snr,'--checkpoint',ck,'--input',root/f'{split}.wav','--output',out]);scores[name][split]=json.loads(out.read_text())
        rows=[json.loads(l) for l in (case/'losses.jsonl').read_text().splitlines()]
        oldrows=[json.loads(l) for l in Path(f'runs/phase3-v1/preconditioned-{seed}/losses.jsonl').read_text().splitlines()]
        assert [(r['step'],r['task'],r['t'],r['rng_state']) for r in rows]==[(r['step'],r['task'],r['t'],r['rng_state']) for r in oldrows]
        cases.append({'seed':seed,'metrics':json.loads((case/'metrics.json').read_text()),'signal':json.loads((case/'signal.json').read_text()),'common_corruption':scores})
    assert sha(source)==plan['source_sha256'] and sha(train)==plan['train_sha256']
    (root/'results.json').write_text(json.dumps({'plan':plan,'cases':cases,'paired_rng_traces_equal':True,'sources_unchanged':True},indent=2)+'\n')
    print('COMPLETE '+str(root/'results.json'),flush=True)


if __name__=='__main__':main()
