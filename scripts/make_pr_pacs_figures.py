#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
STAT = ROOT / "experiments" / "pr_rescue" / "statistics"
OUT = ROOT / "manuscript" / "pr_submission" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

COL_GLOBAL = "#7A7A7A"
COL_PACS = "#3B82A0"
COL_POS = "#2F8F6B"
COL_NEG = "#B85C5C"
COL_ACC = "#D9A441"
COL_LIGHT = "#E9EEF2"


def save(fig, stem, dpi=600):
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax, label):
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def method_schematic():
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.03, 0.22, 0.20, 0.58, "Expert panel", "Repeated labels\n→ empirical mass $\\lambda_x$"),
        (0.29, 0.22, 0.20, 0.58, "Oracle demand", "Model ranking + $q$\n→ minimum top-$k$ $K_q$"),
        (0.55, 0.22, 0.20, 0.58, "Demand model", "Predict $\\widehat{k}(x)$\nfrom model-output features"),
        (0.81, 0.22, 0.16, 0.58, "Conformal correction", "One global one-sided\ncalibration residual\n→ final top-$k$ set"),
    ]
    for i, (x,y,w,h,title,body) in enumerate(boxes):
        fc = COL_LIGHT if i != 3 else "#E6F0F4"
        patch = FancyBboxPatch((x,y), w,h, boxstyle="round,pad=0.012,rounding_size=0.02",
                               facecolor=fc, edgecolor="#AAB4BB", linewidth=1.0)
        ax.add_patch(patch)
        ax.text(x+w/2, y+h*0.72, title, ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(x+w/2, y+h*0.40, body, ha="center", va="center", fontsize=8, linespacing=1.25)
    for x0, x1 in [(0.235,0.285),(0.495,0.545),(0.755,0.805)]:
        ax.add_patch(FancyArrowPatch((x0,0.51),(x1,0.51),arrowstyle="-|>",mutation_scale=12,
                                     linewidth=1.1,color="#69757D"))
    ax.text(0.50, 0.06, "PACS is a post-hoc wrapper: expert labels are used only before deployment; test-time input is the base model output.",
            ha="center", va="center", fontsize=7.5)
    save(fig, "fig1_pacs_method")


def load_stats():
    eff = pd.read_csv(STAT / "paired_effects_main_pacs.csv")
    units = pd.read_csv(STAT / "paired_split_units_main_pacs.csv")
    return eff, units


def fig2_primary(eff, units):
    fig = plt.figure(figsize=(7.1, 4.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 1.0], hspace=0.48, wspace=0.38)

    # a: paired P90 dermatology q=.9
    ax = fig.add_subplot(gs[:,0])
    d = units[(units.dataset=="Dermatology") & (units.q==0.9)].copy()
    x0, x1 = 0, 1
    for _, r in d.iterrows():
        ax.plot([x0,x1], [r.global_p90_size, r.pacs_p90_size], color="#B7C0C6", linewidth=1.0, zorder=1)
        ax.scatter(x0, r.global_p90_size, color=COL_GLOBAL, s=24, zorder=2)
        ax.scatter(x1, r.pacs_p90_size, color=COL_PACS, s=24, zorder=2)
    ax.set_xlim(-0.35,1.35)
    ax.set_xticks([0,1], ["Global panel","PACS"])
    ax.set_ylabel("P90 prediction-set size")
    ax.set_title("Dermatology, $q=0.9$: paired split-level tail burden", loc="left", fontsize=9)
    row = eff[(eff.dataset=="Dermatology")&(eff.q==0.9)&(eff.metric=="p90_size")].iloc[0]
    ax.text(0.03,0.98,f"Δ = {row.delta:.2f} classes\n95% CI [{row.ci_lo:.2f}, {row.ci_hi:.2f}]\nPACS lower in 12/12 splits",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#D5DADD", lw=0.8))
    panel_label(ax,"a")

    # b: success delta CI dermatology q=.9
    axb = fig.add_subplot(gs[0,1])
    sub = eff[(eff.dataset=="Dermatology")&(eff.q.isin([0.7,0.8,0.9]))&(eff.metric=="success")].sort_values("q")
    y = np.arange(len(sub))
    vals = sub.delta.to_numpy()*100
    lo = sub.ci_lo.to_numpy()*100
    hi = sub.ci_hi.to_numpy()*100
    axb.errorbar(vals, y, xerr=[vals-lo,hi-vals], fmt="o", color=COL_PACS, ecolor=COL_PACS, capsize=3, ms=5)
    axb.axvline(0,color="#9CA3A8",lw=0.9)
    axb.set_yticks(y,[f"q={q:.1f}" for q in sub.q])
    axb.set_xlabel("PACS − Global success (percentage points)")
    axb.set_title("Reliability change", loc="left", fontsize=9)
    panel_label(axb,"b")

    # c: p90 delta CI dermatology qs
    axc = fig.add_subplot(gs[1,1])
    sub = eff[(eff.dataset=="Dermatology")&(eff.q.isin([0.7,0.8,0.9]))&(eff.metric=="p90_size")].sort_values("q")
    y = np.arange(len(sub))
    vals = sub.delta.to_numpy()
    lo = sub.ci_lo.to_numpy(); hi=sub.ci_hi.to_numpy()
    colors=[COL_NEG if v>0 else COL_POS for v in vals]
    for yy,v,l,h,c in zip(y,vals,lo,hi,colors):
        axc.errorbar(v,yy,xerr=[[v-l],[h-v]],fmt="o",color=c,ecolor=c,capsize=3,ms=5)
    axc.axvline(0,color="#9CA3A8",lw=0.9)
    axc.set_yticks(y,[f"q={q:.1f}" for q in sub.q])
    axc.set_xlabel("PACS − Global P90 size (classes)")
    axc.set_title("Tail-efficiency change", loc="left", fontsize=9)
    panel_label(axc,"c")

    fig.suptitle("Primary evidence: PACS reduces tail inflation only in the demanding dermatology regime", fontsize=10, y=0.995)
    save(fig,"fig2_dermatology_primary")


def fig3_regime(eff):
    fig, axes = plt.subplots(1,3,figsize=(7.2,2.55))
    # NIH success
    nihs = eff[(eff.dataset=="NIH")&(eff.metric=="success")].sort_values("q")
    x = nihs.q.to_numpy(); vals=nihs.delta.to_numpy()*100
    lo=nihs.ci_lo.to_numpy()*100; hi=nihs.ci_hi.to_numpy()*100
    axes[0].errorbar(x,vals,yerr=[vals-lo,hi-vals],fmt="o-",color=COL_PACS,capsize=3,lw=1.2)
    axes[0].axhline(0,color="#9CA3A8",lw=0.8)
    axes[0].set_xticks([0.7,0.8,0.9])
    axes[0].set_xlabel("Expert-mass requirement $q$")
    axes[0].set_ylabel("Success change (pp)")
    axes[0].set_title("NIH reliability",loc="left",fontsize=9)
    panel_label(axes[0],"a")

    nihp = eff[(eff.dataset=="NIH")&(eff.metric=="p90_size")].sort_values("q")
    vals=nihp.delta.to_numpy(); lo=nihp.ci_lo.to_numpy(); hi=nihp.ci_hi.to_numpy()
    axes[1].errorbar(x,vals,yerr=[vals-lo,hi-vals],fmt="o-",color=COL_PACS,capsize=3,lw=1.2)
    axes[1].axhline(0,color="#9CA3A8",lw=0.8)
    axes[1].set_xticks([0.7,0.8,0.9])
    axes[1].set_xlabel("Expert-mass requirement $q$")
    axes[1].set_ylabel("P90-size change (classes)")
    axes[1].set_title("NIH tail burden",loc="left",fontsize=9)
    panel_label(axes[1],"b")

    # regime map
    ax=axes[2]
    combos=[]
    for ds,q in [("Dermatology",0.7),("Dermatology",0.8),("Dermatology",0.9),("NIH",0.7),("NIH",0.8),("NIH",0.9),("CIFAR-10H",0.7)]:
        s=eff[(eff.dataset==ds)&(eff.q==q)&(eff.metric=="success")].iloc[0]
        p=eff[(eff.dataset==ds)&(eff.q==q)&(eff.metric=="p90_size")].iloc[0]
        combos.append((ds,q,s.delta*100,p.delta))
    marks={"Dermatology":"o","NIH":"s","CIFAR-10H":"^"}
    for ds,q,sd,pd in combos:
        c=COL_POS if pd<0 else COL_NEG
        ax.scatter(pd,sd,s=38,marker=marks[ds],color=c,edgecolor="white",linewidth=0.5,zorder=3)
        ax.text(pd,sd,f" {ds.split('-')[0]} {q:.1f}",fontsize=6.5,va="center")
    ax.axvline(0,color="#9CA3A8",lw=0.8); ax.axhline(0,color="#9CA3A8",lw=0.8)
    ax.set_xlabel("P90-size change (PACS − Global)")
    ax.set_ylabel("Success change (pp)")
    ax.set_title("Operating-regime map",loc="left",fontsize=9)
    panel_label(ax,"c")
    fig.suptitle("Validation and boundary: benefit depends on available cardinality headroom",fontsize=10,y=1.02)
    save(fig,"fig3_regime_map")


def main():
    method_schematic()
    eff, units = load_stats()
    fig2_primary(eff, units)
    fig3_regime(eff)
    print("Wrote figures to", OUT)

if __name__ == "__main__":
    main()
