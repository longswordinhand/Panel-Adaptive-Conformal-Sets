#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans','sans-serif'],
    'svg.fonttype':'none','pdf.fonttype':42,'font.size':7,
    'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':0.8,
    'legend.frameon':False,
})
OUT=Path('manuscript/latex/figures'); OUT.mkdir(parents=True,exist_ok=True)

def save(fig,name):
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight')
    plt.close(fig)

# Fig 1: NIH reader-count contraction
p=Path('experiments/pilot/results/tail_identification/identified_tail_bounds.csv')
df=pd.read_csv(p)
sel=['Abnormal','Consolidation','Pleural Thickening','Nodule','Pneumothorax','Cardiomegaly']
z=df[(df.dataset.eq('NIH-all-findings')) & (df.beta.eq(0.5)) & df.label.isin(sel)].copy()
fig,ax=plt.subplots(figsize=(3.45,2.55))
for label in sel:
    q=z[z.label.eq(label)].sort_values('m')
    ax.plot(q.m,q.width,marker='o',linewidth=1.2,markersize=3,label=label)
ax.set_xlabel('Readers per case, $m$'); ax.set_ylabel('Sharp identified width')
ax.set_xticks([1,2,3,4,5]); ax.set_ylim(bottom=0)
ax.legend(fontsize=5.8,ncol=2,loc='upper right')
ax.text(-0.12,1.03,'a',transform=ax.transAxes,fontweight='bold',fontsize=9)
save(fig,'fig1_nih_reader_contraction')

# Fig 2: honest intervals, separate panels NIH/VinDr
r=pd.read_csv('experiments/pilot/results/tail_honest_ci/real_data_honest_ci_sensitivity.csv')
r=r[(r.beta.eq(0.5)) & r.assumption.isin(['unrestricted','margin_g1_C2']) & r.feasible]
fig,axes=plt.subplots(1,2,figsize=(7.0,2.7),sharex=True)
for ax,dataset,title in zip(axes,['NIH','VinDr'],['NIH: five readers','VinDr-CXR: three readers']):
    q=r[r.dataset.eq(dataset)].copy()
    labels=list(dict.fromkeys(q.label.tolist()))
    y=np.arange(len(labels))
    for j,(assump,mark) in enumerate([('unrestricted','o'),('margin_g1_C2','s')]):
        qq=q[q.assumption.eq(assump)].set_index('label').reindex(labels)
        off=(-0.10 if j==0 else 0.10)
        mid=(qq.lower.values+qq.upper.values)/2
        err=np.vstack([mid-qq.lower.values,qq.upper.values-mid])
        ax.errorbar(mid,y+off,xerr=err,fmt=mark,markersize=3,capsize=2,linewidth=0.9,label=('Unrestricted' if j==0 else r'Margin $\gamma=1,C=2$'))
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis(); ax.set_xlim(0,0.9)
    ax.set_xlabel(r'95% confidence interval for $\tau_{0.5}$'); ax.set_title(title,fontsize=7.5)
    ax.grid(axis='x',linewidth=0.3,alpha=0.35)
axes[0].set_ylabel('Finding')
axes[0].text(-0.25,1.04,'a',transform=axes[0].transAxes,fontweight='bold',fontsize=9)
axes[1].text(-0.20,1.04,'b',transform=axes[1].transAxes,fontweight='bold',fontsize=9)
axes[0].legend(fontsize=5.8,loc='lower right')
save(fig,'fig2_honest_ci_forest')

# Fig 3: synthetic population widths
s=pd.read_csv('experiments/pilot/results/tail_identification/synthetic_population_bounds.csv')
name_map={'beta_2_5':'Beta(2,5)','beta_5_2':'Beta(5,2)','bimodal_2_8__8_2':'Bimodal','near_threshold_20_20':'Beta(20,20), near threshold','asymmetric_mix':'Asymmetric mixture'}
fig,ax=plt.subplots(figsize=(3.45,2.55))
for key,label in name_map.items():
    q=s[s.scenario.eq(key)].sort_values('m')
    if len(q): ax.plot(q.m,q.width,marker='o',linewidth=1.2,markersize=3,label=label)
ax.set_xscale('log'); ax.set_xticks([1,2,3,5,8,12,20]); ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax.set_xlabel('Readers per case, $m$'); ax.set_ylabel('Sharp identified width')
ax.set_ylim(0,1.03); ax.legend(fontsize=5.6,loc='upper right')
ax.text(-0.12,1.03,'a',transform=ax.transAxes,fontweight='bold',fontsize=9)
save(fig,'fig3_synthetic_threshold_mass')
print('generated', len(list(OUT.glob('fig*.pdf'))), 'PDF figures')
