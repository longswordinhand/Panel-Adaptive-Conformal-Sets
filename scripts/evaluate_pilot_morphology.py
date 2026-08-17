#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from multirater_conformal_seg.calibration.pixel_interval_cp import consensus_mask
from multirater_conformal_seg.calibration.morphological_band_cp import precompute_geometry,inclusion_radius,band_metrics,split_quantile
CACHE=ROOT/'data/processed/prostate_512'; SPLITS=ROOT/'experiments/pilot/splits/prostate_5fold_patient_splits.csv'; PRED=ROOT/'experiments/pilot/predictions/main'; AUDIT=ROOT/'data/processed/audit_prostate/case_task_audit.csv'; OUT=ROOT/'experiments/pilot/results/morph_v2'
ALPHAS=(.1,.2,.3); REPS=200; SEED=20260815; EXT={'case07','case50'}
def load(fold,task,c):
 p=np.load(PRED/f'fold{fold}'/f'task{task}'/f'{c}.npy').astype(np.float32); g=precompute_geometry(p)
 with np.load(CACHE/f'{c}.npz') as d: ms=d[f'task{task}_masks'].astype(np.uint8)
 sc=np.array([inclusion_radius(g,m) for m in ms],float); cons=consensus_mask([m for m in ms]); cs=inclusion_radius(g,cons)
 return {'g':g,'scores':sc,'cons_score':cs,'cons_area':int(cons.sum())}
def metr(c,r):
 cov=c['scores']<=r+1e-9; return {'random_rater_coverage':float(cov.mean()),'mean_expert_miss_rate':float(1-cov.mean()),'all_rater_coverage':float(cov.all()),'consensus_coverage':float(c['cons_score']<=r+1e-9),**band_metrics(c['g'],r,c['cons_area'])}
def main():
 OUT.mkdir(parents=True,exist_ok=True); s=pd.read_csv(SPLITS); a=pd.read_csv(AUDIT); a['task']=a.task.astype(int).map(lambda x:f'{x:02d}')
 rows=[]; th=[]; req=[]
 for task in ['01','02']:
  for fold in range(5):
   f=s[s.fold==fold]; calids=sorted(f[f.role=='calibration'].case_id); testids=sorted(f[f.role=='test'].case_id)
   cal={c:load(fold,task,c) for c in calids}; test={c:load(fold,task,c) for c in testids}
   for c,cc in test.items():
    r=float(cc['scores'].max()); ar=a[(a.task==task)&(a.case_id==c)].iloc[0]; bm=band_metrics(cc['g'],r,cc['cons_area'])
    req.append({'task':task,'fold':fold,'case_id':c,'r_required_all':r,'required_all_ambiguity_fraction':bm['ambiguity_fraction_image'],'required_all_ambiguity_ratio':bm['ambiguity_to_consensus_ratio'],'audit_disagreement_1_minus_min_dice':float(1-ar.pairwise_dice_min),'extreme_task02':task=='02' and c in EXT})
   cs=[cal[c]['cons_score'] for c in calids]; ns=[x for c in calids for x in cal[c]['scores']]; als=[float(cal[c]['scores'].max()) for c in calids]
   for alpha in ALPHAS:
    qs={'consensus':split_quantile(cs,alpha),'naive_annotation':split_quantile(ns,alpha),'all_rater':split_quantile(als,alpha)}
    for m,r in qs.items():
     th.append({'task':task,'fold':fold,'alpha':alpha,'method':m,'replicate':-1,'radius':r})
     for c,cc in test.items(): rows.append({'task':task,'fold':fold,'alpha':alpha,'method':m,'replicate':-1,'case_id':c,'radius':r,'extreme_task02':task=='02' and c in EXT,**metr(cc,r)})
    for rep in range(REPS):
     rng=np.random.default_rng(SEED+int(task)*100000+fold*1000+int(alpha*100)*10+rep); ss=[cal[c]['scores'][int(rng.integers(len(cal[c]['scores'])))] for c in calids]; r=split_quantile(ss,alpha)
     th.append({'task':task,'fold':fold,'alpha':alpha,'method':'random_rater','replicate':rep,'radius':r})
     for c,cc in test.items(): rows.append({'task':task,'fold':fold,'alpha':alpha,'method':'random_rater','replicate':rep,'case_id':c,'radius':r,'extreme_task02':task=='02' and c in EXT,**metr(cc,r)})
 d=pd.DataFrame(rows); t=pd.DataFrame(th); q=pd.DataFrame(req); d.to_csv(OUT/'case_level_metrics.csv',index=False); t.to_csv(OUT/'calibration_thresholds.csv',index=False); q.to_csv(OUT/'required_radius_by_case.csv',index=False)
 det=d[d.method!='random_rater']; rr=d[d.method=='random_rater'].groupby(['task','fold','alpha','method','case_id','extreme_task02'],as_index=False).mean(numeric_only=True); comb=pd.concat([det,rr],ignore_index=True); comb.to_csv(OUT/'case_level_metrics_replicate_averaged.csv',index=False)
 mets=['random_rater_coverage','mean_expert_miss_rate','all_rater_coverage','consensus_coverage','ambiguity_fraction_image','ambiguity_to_consensus_ratio','normalized_radius']; out=[]
 for (task,alpha,m),g in comb.groupby(['task','alpha','method']):
  rec={'task':task,'alpha':alpha,'method':m,'n_cases':len(g),**{x:float(g[x].mean()) for x in mets}}; out.append(rec)
  if task=='02':
   z=g[~g.extreme_task02]; out.append({'task':task,'alpha':alpha,'method':m+'__sensitivity_no_case07_case50','n_cases':len(z),**{x:float(z[x].mean()) for x in mets}})
 sm=pd.DataFrame(out); sm.to_csv(OUT/'summary_metrics.csv',index=False)
 ass=[]
 for task,g in q.groupby('task'):
  for y in ['r_required_all','required_all_ambiguity_fraction','required_all_ambiguity_ratio']:
   x=g.audit_disagreement_1_minus_min_dice; pr=pearsonr(x,g[y]); sr=spearmanr(x,g[y]); ass.append({'task':task,'outcome':y,'n':len(g),'pearson_r':pr.statistic,'pearson_p':pr.pvalue,'spearman_rho':sr.statistic,'spearman_p':sr.pvalue})
   if task=='02':
    z=g[~g.extreme_task02]; pr=pearsonr(z.audit_disagreement_1_minus_min_dice,z[y]); sr=spearmanr(z.audit_disagreement_1_minus_min_dice,z[y]); ass.append({'task':task,'outcome':y+'__sensitivity_no_case07_case50','n':len(z),'pearson_r':pr.statistic,'pearson_p':pr.pvalue,'spearman_rho':sr.statistic,'spearman_p':sr.pvalue})
 ad=pd.DataFrame(ass); ad.to_csv(OUT/'disagreement_associations.csv',index=False); (OUT/'integrity.json').write_text(json.dumps({'folds':5,'tasks':['01','02'],'alphas':ALPHAS,'random_reps':REPS,'rows':len(d)},indent=2))
 print('SUMMARY\n',sm.to_string(index=False)); print('\nASSOCIATIONS\n',ad.to_string(index=False))
if __name__=='__main__': main()
