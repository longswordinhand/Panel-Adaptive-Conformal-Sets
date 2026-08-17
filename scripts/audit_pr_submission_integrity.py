#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, math, sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
facts_path = ROOT/'manuscript/results_facts.json'
facts = json.loads(facts_path.read_text())
errors=[]; checks=[]

def sha256(p: Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def close(name,a,b,tol=5e-4):
    ok=math.isclose(float(a),float(b),abs_tol=tol,rel_tol=0)
    checks.append((name,ok,float(a),float(b)))
    if not ok: errors.append(f'{name}: got {a}, expected {b}')

# Hash checks for every frozen source.
for key,meta in facts['sources'].items():
    p=ROOT/meta['path']
    if not p.exists():
        errors.append(f'missing source {key}: {p}')
        continue
    got=sha256(p); exp=meta['sha256']; ok=(got==exp)
    checks.append((f'hash:{key}',ok,got,exp))
    if not ok: errors.append(f'hash mismatch {key}: {got} != {exp}')

# Dermatology q=.9: raw ablation file, means over all model-by-split rows.
d=pd.read_csv(ROOT/facts['sources']['dermatology_q09_ablation_raw']['path'])
for method,prefix in [('global_panel','q09_global'),('pacs_global_residual','q09_pacs')]:
    z=d[d.method==method]
    close(f'derm:{method}:success', z.case_mass_success.mean(), facts['key_facts']['dermatology'][prefix+'_success'])
    close(f'derm:{method}:mean_size', z.mean_size.mean(), facts['key_facts']['dermatology'][prefix+'_mean_size'], 5e-3)
    close(f'derm:{method}:p90', z.p90_size.mean(), facts['key_facts']['dermatology'][prefix+'_p90'], 5e-3)

# NIH final PACS raw + baseline raw.
npacs=pd.read_csv(ROOT/facts['sources']['nih_global_residual_raw']['path'])
for q,tag in [(0.7,'q07'),(0.8,'q08'),(0.9,'q09')]:
    z=npacs[np.isclose(npacs.q,q)]
    close(f'nih:pacs:{q}:success',z.success.mean(),facts['key_facts']['nih'][tag+'_pacs_success'])
    if tag in ('q07','q08'):
        close(f'nih:pacs:{q}:mean_size',z.mean_size.mean(),facts['key_facts']['nih'][tag+'_pacs_mean_size'],5e-3)
    close(f'nih:pacs:{q}:p90',z.p90_size.mean(),facts['key_facts']['nih'][tag+'_pacs_p90'],5e-3)
    b=pd.read_csv(ROOT/facts['sources'][f'nih_global_{tag}_baseline_raw']['path'])
    g=b[b.method=='global_panel']
    close(f'nih:global:{q}:success',g.success.mean(),facts['key_facts']['nih'][tag+'_global_success'])
    close(f'nih:global:{q}:p90',g.p90_size.mean(),facts['key_facts']['nih'][tag+'_global_p90'],5e-3)

# CIFAR final PACS + global baseline.
cp=pd.read_csv(ROOT/facts['sources']['cifar_global_residual_raw']['path'])
cb=pd.read_csv(ROOT/facts['sources']['cifar_global_baseline_raw']['path'])
g=cb[cb.method=='global_panel']
for name,series,expected,tol in [
 ('cifar:pacs:success',cp.success.mean(),facts['key_facts']['cifar10h']['q07_pacs_success'],5e-4),
 ('cifar:pacs:mean_size',cp.mean_size.mean(),facts['key_facts']['cifar10h']['q07_pacs_mean_size'],5e-3),
 ('cifar:pacs:p90',cp.p90_size.mean(),facts['key_facts']['cifar10h']['q07_pacs_p90'],5e-3),
 ('cifar:global:success',g.success.mean(),facts['key_facts']['cifar10h']['q07_global_success'],5e-4),
 ('cifar:global:mean_size',g.mean_size.mean(),facts['key_facts']['cifar10h']['q07_global_mean_size'],5e-3),
 ('cifar:global:p90',g.p90_size.mean(),facts['key_facts']['cifar10h']['q07_global_p90'],5e-3)]: close(name,series,expected,tol)

# Paired headline effect and CI.
pair=pd.read_csv(ROOT/facts['sources']['paired_effects']['path'])
r=pair[(pair.dataset=='Dermatology') & (pair.q==0.9) & (pair.metric=='p90_size')].iloc[0]
close('paired:derm:q09:p90:delta',r.delta,facts['key_facts']['dermatology']['q09_p90_delta'],5e-3)
close('paired:derm:q09:p90:ci_lo',r.ci_lo,facts['key_facts']['dermatology']['q09_p90_delta_bootstrap95'][0],5e-3)
close('paired:derm:q09:p90:ci_hi',r.ci_hi,facts['key_facts']['dermatology']['q09_p90_delta_bootstrap95'][1],5e-3)

print(f'checks={len(checks)} failures={len(errors)}')
for name,ok,got,exp in checks:
    print(('PASS' if ok else 'FAIL'),name,got,exp)
if errors:
    print('\nFAILURES:',file=sys.stderr)
    for e in errors: print('-',e,file=sys.stderr)
    sys.exit(1)
print('INTEGRITY_AUDIT_PASS')
