#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / 'paper_project'
SRC = ROOT / 'results_paperlock' / 'tables'
main = pd.read_csv(SRC/'table_main_results_paperlock.csv')
idx = {r['Method']: r for _, r in main.iterrows()}

def imp(prop, base, metric, direction):
    p=float(idx[prop][metric]); b=float(idx[base][metric])
    if abs(b) < 1e-12:
        return None
    return 100*(b-p)/b if direction=='min' else 100*(p-b)/b
checks = {
    ('Reactive-3C','Objective','min'): 6.47,
    ('Reactive-3C','Avg Delay','min'): 0.61,
    ('Reactive-3C','p95 Delay','min'): 2.14,
    ('Reactive-3C','Energy','min'): -0.72,
    ('Reactive-3C','Handovers/100','min'): 24.52,
    ('Reactive-3C','Useful Cache Hit','max'): -1.05,
    ('Reactive-3C','Outage','min'): 9.14,
    ('Reactive-3C','Drop Rate','min'): -0.53,
    ('MUCCO-like','Objective','min'): 44.32,
    ('MUCCO-like','Handovers/100','min'): 83.33,
    ('RateMax','Objective','min'): 67.20,
    ('RateMax','Energy','min'): 16.77,
    ('RateMax','Handovers/100','min'): 91.52,
    ('RateMax','Drop Rate','min'): 11.37,
}
fail=[]
for (base,metric,direction), expected in checks.items():
    val = imp('P3C-LR', base, metric, direction)
    if val is None or abs(round(val,2)-expected) > 0.02:
        fail.append((base,metric,val,expected))
if fail:
    raise SystemExit('Number verification failed: '+repr(fail))

paper = (PROJECT/'sections'/'08_results.tex').read_text(encoding='utf-8') + '\n' + (PROJECT/'sections'/'00_abstract.tex').read_text(encoding='utf-8')
for needle in ['6.47\\%','0.61\\%','2.14\\%','24.52\\%','9.14\\%','0.72\\%','1.05\\%','0.53\\%']:
    if needle not in paper:
        fail.append(('missing_text', needle))
if fail:
    raise SystemExit('Reported number text check failed: '+repr(fail))
print('Result number verification PASS')
