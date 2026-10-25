"""Read-only Android scheduling/resource observations, with conservative decisions.

cpuset is evidence of Android process class, not proof of screen visibility.
No clocks, priorities, affinity, system files or thermal controls are modified.
"""
from pathlib import Path
import os


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def snapshot():
    cpuset = read('/proc/self/cpuset')
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else []
    if cpuset and ('background' in cpuset or 'restricted' in cpuset):
        state = 'background'
    elif cpuset and ('top-app' in cpuset or 'foreground' in cpuset):
        state = 'foreground-class'
    elif cpuset and ('moderate' in cpuset or (affinity and len(affinity) <= 4)):
        state = 'reduced-class'
    else:
        state = 'unknown'
    temperatures = {}
    for path in Path('/sys/class/thermal').glob('thermal_zone*'):
        name = read(path/'type')
        value = read(path/'temp')
        if name and value and ('gpu' in name or name == 'battery'):
            try:
                degrees = float(value)/1000
                if 0 < degrees < 200:
                    temperatures[name] = max(degrees, temperatures.get(name, degrees))
            except ValueError:
                pass
    memory = {}
    for line in (read('/proc/meminfo') or '').splitlines():
        parts = line.split()
        if parts[0] in ['MemAvailable:', 'MemTotal:']:
            memory[parts[0][:-1]+'_kib'] = int(parts[1])
    gpu = max((v for k,v in temperatures.items() if 'gpu' in k), default=None)
    battery = temperatures.get('battery')
    reasons = []
    if gpu is None:
        reasons.append('GPU temperature unavailable')
    elif gpu >= 65:
        reasons.append('GPU operator threshold 65C reached')
    if battery is not None and battery >= 41:
        reasons.append('battery operator threshold 41C reached')
    if memory.get('MemAvailable_kib',0) < 524288:
        reasons.append('available RAM below512MiB')
    return {'cpuset':cpuset,'scheduling_state':state,'affinity':affinity,
            'temperatures_c':temperatures,'memory':memory,'allow_training':not reasons,
            'defer_reasons':reasons,'operator_limits':{'gpu_c':65,'battery_c':41,'minimum_available_kib':524288},'chunk_steps':2000 if state=='foreground-class' and len(affinity)>=6 else 500,
            'build_jobs':4 if state=='foreground-class' and len(affinity)>=6 else 2}


if __name__ == '__main__':
    import json
    print(json.dumps(snapshot(),indent=2))
