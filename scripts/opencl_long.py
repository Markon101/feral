#!/usr/bin/env python3
"""Checkpointed sequential OpenCL continuation with between-chunk cooling checks.

Thresholds are conservative operator settings, not manufacturer safety ratings.
No drivers, system settings, model architecture, data or learning rate are changed.
"""
import argparse,datetime,hashlib,json,os,subprocess,time
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(65536),b''):h.update(b)
    return h.hexdigest()


def sensors():
    values={}
    for p in Path('/sys/class/thermal').glob('thermal_zone*'):
        try:
            name=(p/'type').read_text().strip()
            if 'gpu' in name or name=='battery':
                v=float((p/'temp').read_text().strip())/1000
                if 0<v<200:values[name]=max(v,values.get(name,v))
        except (OSError,ValueError):pass
    return values


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--total-steps',type=int,default=100000)
    p.add_argument('--chunk-steps',type=int,default=2000)
    a=p.parse_args();root=a.run.resolve();root.mkdir(parents=True,exist_ok=False)
    checkpoint=a.checkpoint.resolve(strict=True);c=json.loads(checkpoint.read_text());done=c['completed_steps']
    assert done<a.total_steps<=200000 and 1<=a.chunk_steps<=5000
    source=Path(c['dataset']['sources'][0]['path']);assert sha(source)==c['dataset']['sources'][0]['sha256']
    assert c['flow']=={'kind':'preconditioned','data_scale':c['flow']['data_scale'],'prior_std':c['flow']['prior_std']}
    launcher=Path('scripts/feral-opencl.sh').resolve(strict=True)
    plan={'initial_checkpoint':str(checkpoint),'initial_sha256':sha(checkpoint),'initial_steps':done,'target_total_steps':a.total_steps,'chunk_steps':a.chunk_steps,'source':str(source),'source_sha256':sha(source),'foreground':'owner confirmed, phone initially warm','gpu_pause_c':60,'gpu_resume_c':55,'battery_pause_c':41,'battery_resume_c':39,'minimum_available_memory_kib':524288,'temperature_scope':'read before and after chunks; no claim about unobserved in-chunk peaks','commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');chunks=[];cooling=0.;index=0
    def record(event):
        with (root/'events.jsonl').open('a') as f:f.write(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**event})+'\n')
    def memory():
        for l in Path('/proc/meminfo').read_text().splitlines():
            if l.startswith('MemAvailable:'):return int(l.split()[1])
        return 0
    while done<a.total_steps:
        values=sensors();gpu=max((v for k,v in values.items() if 'gpu' in k),default=None);battery=values.get('battery')
        if gpu is None:raise RuntimeError('GPU temperature monitoring unavailable; checkpoint preserved, refusing an unmonitored long run')
        hot=gpu>=60 or (battery is not None and battery>=41) or memory()<524288
        if hot:
            print('Cooling before next chunk:',values,flush=True);wait_start=time.monotonic()
            while hot:
                record({'event':'cooling','temperatures_c':values,'memory_available_kib':memory(),'completed_steps':done})
                if time.monotonic()-wait_start>300:raise RuntimeError('cooling/memory condition did not recover in five minutes; latest checkpoint preserved')
                time.sleep(5);values=sensors();gpu=max((v for k,v in values.items() if 'gpu' in k),default=999);battery=values.get('battery');hot=gpu>55 or (battery is not None and battery>39) or memory()<524288
            cooling+=time.monotonic()-wait_start
        steps=min(a.chunk_steps,a.total_steps-done);folder=root/f'chunk-{index:03d}'
        argv=[str(launcher),'train','--backend','opencl','--input',str(source),'--run',str(folder),'--network','pyramid','--parameterization','preconditioned','--data-scale',str(c['flow']['data_scale']),'--prior-std',str(c['flow']['prior_std']),'--steps',str(steps),'--seed',str(c['initial_seed']),'--no-normalize','--resume',str(checkpoint),'--preview-every',str(steps),'--checkpoint-every',str(min(1000,steps))]
        record({'event':'start_chunk','argv':argv,'temperatures_c':values,'completed_steps':done});print(f'OpenCL {done} -> {done+steps} | {values}',flush=True)
        now=time.monotonic()
        with (root/f'chunk-{index:03d}.log').open('x') as log:subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,check=True)
        checkpoint=folder/'checkpoint.json';state=json.loads(checkpoint.read_text());assert state['completed_steps']==done+steps and state['adam']['step']==done+steps
        metadata=json.loads((folder/'run.json').read_text());assert metadata['backend']=='opencl-f32-resident'
        assert state['dataset']==c['dataset'] and state['flow']==c['flow']
        metrics=json.loads((folder/'metrics.json').read_text());done+=steps
        entry={'chunk':index,'steps':steps,'total_steps':done,'wall_seconds':time.monotonic()-now,'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'metrics':metrics,'temperatures_c':sensors()};chunks.append(entry);record({'event':'end_chunk',**entry})
        index+=1
    assert sha(source)==plan['source_sha256']
    (root/'summary.json').write_text(json.dumps({'plan':plan,'chunks':chunks,'final_checkpoint':str(checkpoint),'final_steps':done,'cooling_seconds':cooling,'source_unchanged':True,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
    print('COMPLETE',checkpoint,flush=True)


if __name__=='__main__':main()
