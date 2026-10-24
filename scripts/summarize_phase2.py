#!/usr/bin/env python3
"""Aggregate model-seed results without treating probe crops as replications."""
import argparse
import json
from pathlib import Path
from statistics import mean


def load(path):
    return json.loads(Path(path).read_text())


def probe_summary(evaluation):
    rows = evaluation['rows']
    result = {'velocity_mse': evaluation['mean_velocity_mse'], 'by_time': {}, 'by_task': {}}
    for t in [0.1, 0.5, 0.9]:
        rr = [r for r in rows if abs(r['t']-t) < 1e-6]
        result['by_time'][str(t)] = {mode: {metric: mean(r[mode][metric] for r in rr)
                                                   for metric in ['mse','derivative_mse','log_spectral_l1']}
                                      for mode in ['endpoint','zero_audio','noisy_input']}
    for task in ['unconditional','prefix','hole','suffix']:
        result['by_task'][task] = mean(r['velocity_mse'] for r in rows if r['task']==task)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--budget', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    base, budget = load(a.base/'results.json'), load(a.budget/'results.json')
    result = {'equal_steps': [], 'compute_followup': [],
              'plans': {'equal_steps': base['plan'], 'followup': budget['plan']},
              'paired_rng_traces_equal': base['paired_rng_traces_equal'],
              'source_unchanged': base['source_unchanged'] and budget['source_unchanged']}
    for case in base['cases']:
        result['equal_steps'].append({'network': case['network'], 'seed': case['seed'],
                                     'metrics': case['metrics'], 'signal': case['signal'],
                                     'scores': {sp: probe_summary(v) for sp, v in case['evaluations'].items()}})
    for case in budget['cases']:
        name, seed = case['name'], case['seed']
        entry = {'name': name, 'seed': seed, 'scores': {sp: probe_summary(v) for sp,v in case['evaluations'].items()}}
        if name == 'pyramid1200':
            prior = load(a.base/f'pyramid-{seed}/metrics.json')
            current = load(a.budget/f'pyramid-{seed}/metrics.json')
            entry['total_update_seconds'] = (prior['steps']*prior['step_mean_ms'] + current['steps']*current['step_mean_ms'])/1000
            entry['peak_trainer_rss_kib'] = max(prior['peak_rss_kib'], current['peak_rss_kib'])
            entry['final_sampling_ms'] = current['final_sampling_ms']
            entry['signal'] = load(a.budget/f'pyramid-{seed}/signal.json')
        elif name == 'baseline600':
            m = load(a.base/f'baseline-{seed}/metrics.json')
            entry['total_update_seconds'] = m['steps']*m['step_mean_ms']/1000
        result['compute_followup'].append(entry)
    result['means'] = {
        'equal_step_test': {n: mean(c['scores']['test']['velocity_mse'] for c in result['equal_steps'] if c['network']==n) for n in ['baseline','pyramid']},
        'fresh_test': {n: mean(c['scores']['test']['velocity_mse'] for c in result['compute_followup'] if c['name']==n) for n in ['baseline600','pyramid600','pyramid1200']}}
    result['limits'] = 'three model seeds, one song; follow-up has twice the pooled updates; descriptors and denoising probes do not establish perceptual quality or rule out transformed memorization'
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result['means'], indent=2))


if __name__ == '__main__':
    main()
