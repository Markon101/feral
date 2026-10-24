#!/usr/bin/env python3
"""Export matched 10-second neural listening examples and labeled non-neural controls.
Run from the repository root after building feral and the phase3_controls example.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import wave

from verify_audio import read_float_wav


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--initialized', type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--training-reference', required=True, type=Path)
    parser.add_argument('--source-offset', type=float, default=80)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run', required=True, type=Path)
    a = parser.parse_args()
    if not 0 <= a.seed < 2**64-32:
        parser.error('--seed must be in 0..2^64-33')
    ck, baseline = a.checkpoint.resolve(strict=True), a.baseline.resolve(strict=True)
    root = a.run.resolve()
    root.mkdir(parents=True, exist_ok=False)
    raw, out = root/'raw', root/'exports'
    meta, matched, comp = out/'metadata', out/'level_matched', out/'comparisons_native'
    for path in [raw, out, meta, matched, comp]:
        path.mkdir()
    binary = Path('target/release/feral').resolve(strict=True)
    controls_binary = Path('target/release/examples/phase3_controls').resolve(strict=True)
    env = os.environ.copy()
    termux_cpp = Path('/data/data/com.termux/files/usr/lib/libc++_shared.so')
    if termux_cpp.exists():
        env['LD_PRELOAD'] = str(termux_cpp) + (':'+env['LD_PRELOAD'] if env.get('LD_PRELOAD') else '')

    def run(argv):
        argv = list(map(str, argv))
        print('RUN ' + ' '.join(argv), flush=True)
        with (root/'commands.jsonl').open('a') as f:
            f.write(json.dumps({'argv': argv})+'\n')
        with (root/'render.log').open('a') as f:
            subprocess.run(argv, check=True, env=env, stdout=f, stderr=subprocess.STDOUT)

    def export(src, dst, gain=None):
        argv = ['ffmpeg', '-nostdin', '-v', 'error', '-n', '-i', src]
        if gain is not None:
            argv += ['-af', f'volume={gain:.12g}']
        run(argv+['-c:a', 'pcm_s16le', dst])
        with wave.open(str(dst), 'rb') as w:
            assert (w.getnframes(), w.getframerate(), w.getsampwidth(), w.getnchannels()) == (160000,16000,2,1)

    run(['ffmpeg','-nostdin','-v','error','-n','-ss',a.source_offset,'-threads','1',
         '-i',a.reference,'-t','10','-map','0:a:0','-ac','1','-ar','16000',
         '-c:a','pcm_f32le',raw/'reference.wav'])
    run(['ffmpeg','-nostdin','-v','error','-n','-i',raw/'reference.wav','-t','5',
         '-c:a','pcm_f32le',raw/'prefix.wav'])
    export(raw/'reference.wav',out/'00_reference.wav')
    cases = [('01_generated',ck,a.seed,'heun',32,1,0),
             ('02_ood_control',ck,a.seed+31,'euler',48,1.4,0),
             ('03_weird',ck,a.seed+31,'euler',48,1.4,.08),
             ('baseline',baseline,a.seed,'heun',32,1,0)]
    if a.initialized:
        cases.append(('initialized',a.initialized.resolve(strict=True),a.seed,'heun',32,1,0))
    for label, checkpoint, seed, solver, steps, noise, perturb in cases:
        flags=['--solver',solver,'--flow-steps',steps,'--noise-scale',noise,'--perturb',perturb]
        run([binary,'generate','--checkpoint',checkpoint,'--seconds',5,'--seed',seed,
             *flags,'--output',raw/f'{label}-first5.wav'])
        run([binary,'continue','--checkpoint',checkpoint,'--input',raw/f'{label}-first5.wav',
             '--seconds',5,'--seed',seed+1,*flags,'--output',raw/f'{label}.wav'])
        export(raw/f'{label}.wav',(comp/f'{label}.wav') if label in ['baseline','initialized'] else out/f'{label}.wav')
    for mode,label,extra,input_file in [
            ('continue','04_continuation',['--seconds',5],raw/'prefix.wav'),
            ('inpaint','05_inpaint',['--start',3,'--end',7],raw/'reference.wav'),
            ('variation','06_variation',[],raw/'reference.wav')]:
        run([binary,mode,'--checkpoint',ck,'--input',input_file,'--seed',a.seed,
             '--solver','heun','--flow-steps',32,*extra,'--output',raw/f'{label}.wav'])
        export(raw/f'{label}.wav',out/f'{label}.wav')
    run([controls_binary,'--reference',a.training_reference,'--output',raw/'controls','--samples',160000,'--seed',a.seed])
    controls={'current':raw/'01_generated.wav','baseline':raw/'baseline.wav',
              'gaussian_control':raw/'controls/gaussian.wav','ar1_control':raw/'controls/ar1.wav',
              'source_reference':raw/'reference.wav'}
    if a.initialized:
        controls['initialized']=raw/'initialized.wav'
    levels={}
    for name,path in controls.items():
        _,samples,_=read_float_wav(path)
        levels[name]={'rms':math.sqrt(sum(v*v for v in samples)/len(samples)),
                      'peak':max(map(abs,samples)),'input':str(path)}
    target=min(.08,*[v['rms']*.97/max(v['peak'],1e-8) for v in levels.values()])
    assert target > 0 and math.isfinite(target)
    for name,path in controls.items():
        if name in ['gaussian_control','ar1_control']:
            export(path,comp/f'{name}.wav')
        gain=target/levels[name]['rms']
        export(path,matched/f'{name}.wav',gain)
        levels[name]['gain']=gain
    _,_,ref=read_float_wav(raw/'reference.wav')
    _,_,continued=read_float_wav(raw/'04_continuation.wav')
    _,_,inpainted=read_float_wav(raw/'05_inpaint.wav')
    assert continued[:320000] == ref[:320000]
    assert inpainted[:192000] == ref[:192000] and inpainted[448000:] == ref[448000:]

    def pcm(path):
        with wave.open(str(path),'rb') as w:
            return w.readframes(w.getnframes())
    ref=pcm(out/'00_reference.wav')
    continued=pcm(out/'04_continuation.wav')
    inpainted=pcm(out/'05_inpaint.wav')
    assert continued[:160000] == ref[:160000]
    assert inpainted[:96000] == ref[:96000] and inpainted[224000:] == ref[224000:]
    for path in raw.rglob('*.json'):
        dst=meta/path.relative_to(raw)
        dst.parent.mkdir(parents=True,exist_ok=True)
        dst.write_text(path.read_text())
    state=json.loads(ck.read_text())
    base_state=json.loads(baseline.read_text())
    guide=f'''FERAL — OpenCL longer-training listening pack

Main WAVs: current checkpoint, {state['completed_steps']} total updates.
16 kHz mono PCM16, exactly 10 seconds each.
00 reference: source seconds {a.source_offset}–{a.source_offset+10}.
01 generated: seeds{a.seed}/{a.seed+1}, Heun32; five seconds from noise plus five seconds learned continuation. Entire clip generated.
02 OOD control: seeds{a.seed+31}/{a.seed+32}, Euler48, noise multiplier1.4, perturbation0.
03 weird: identical to02 plus physical velocity perturbation0.08.
04 continuation: first5seconds source, next5generated.
05 inpaint: source protected outside3–7seconds, generated inside.
06 variation: experimental short protected anchors with learned regeneration between.

Compare level_matched/current.wav and baseline.wav ({base_state['completed_steps']} updates) first.
Initialized, Gaussian and training-fitted AR1 controls are labeled. Gaussian/AR1 are NON-NEURAL controls, never FERAL output.
Level matching uses constant gain only, no filtering/EQ/mastering. Native levels are retained.
The tiny model has at most468ms context and training uses only24seconds from one song. Longer outputs are exploratory; lower noise or loss does not establish music, generalization or absence of memorization.
GPU accelerates training; these WAVs use the portable checkpoint and CPU sampler.
Float originals, commands, solver details and checkpoint hashes remain in the repository.
'''
    (out/'LISTENING_GUIDE.txt').write_text(guide)
    files=[{'file':str(p.relative_to(out)),'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()]
    manifest={'checkpoint':str(ck),'checkpoint_sha256':sha(ck),'training_steps':state['completed_steps'],
              'baseline':str(baseline),'baseline_sha256':sha(baseline),'initialized':str(a.initialized),
              'flow':state['flow'],'seed':a.seed,'level_match_target_rms':target,'level_match_inputs':levels,
              'reference':str(a.reference.resolve()),'reference_sha256':sha(a.reference),
              'files':files,'protected_float_and_pcm16_regions_exact':True}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('COMPLETE '+str(out),flush=True)


if __name__=='__main__':
    main()
