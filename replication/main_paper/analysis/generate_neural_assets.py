"""Generate manuscript assets from released aggregate results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
FINAL=ROOT/'aggregate_results'
DIAG=FINAL/'diagnostics'
FIGURES=ROOT/'figures'
TABLES=ROOT/'tables'
GENERATED=ROOT/'scalars'
for directory in (FIGURES,TABLES,GENERATED): directory.mkdir(parents=True,exist_ok=True)
def nn_metrics(): return pd.read_csv(FINAL/'neural_head_metrics.csv')

import generate_manuscript_figures as old
DATASETS=old.DATASETS
ENC=old.ENCODERS
COMMIT='d3d96e09bb49f33dcf3e5cb0a875ac5d661ae0ad'
fixed=pd.read_csv(FINAL/'fixed_all_methods.csv')
def fmt(x,n=4):return f'{x:.{n}f}'
def appendix_figures(nn):
    f=fixed.set_index(['dataset','method']);n=nn.set_index(['dataset','encoder'])
    matrix=[];lr=[];nr=[]
    short=[v.replace(' semantic','').replace('-B/14','').replace('-B/16','') for v in ENC.values()]
    for d in DATASETS:
        col=old.metric_column(d);asc=d=='facial_age'
        a=pd.Series([f.loc[(d,e),col] for e in ENC]);b=pd.Series([n.loc[(d,e),col] for e in ENC])
        matrix.append((a-b if asc else b-a).to_numpy())
        lr.append(a.rank(ascending=asc,method='average').to_numpy());nr.append(b.rank(ascending=asc,method='average').to_numpy())
    matrix=np.asarray(matrix);scale=np.max(np.abs(matrix),axis=1,keepdims=True)
    norm=np.divide(matrix,scale,out=np.zeros_like(matrix),where=scale>0)
    fig,ax=plt.subplots(figsize=(5.0,3.6))
    im=ax.imshow(norm,aspect='auto',cmap='RdBu_r',vmin=-1,vmax=1)
    ax.set_xticks(range(10),short,rotation=55,ha='right',fontsize=9)
    ax.set_yticks(range(6),[old.NAMES[d] for d in DATASETS],fontsize=9)
    c=fig.colorbar(im,ax=ax,fraction=.04,pad=.025);c.ax.tick_params(labelsize=8)
    c.set_label('Normalized NN improvement',fontsize=9)
    ax.grid(False);fig.tight_layout();fig.savefig(FIGURES/'appendix_nn_heatmap.pdf',bbox_inches='tight');plt.close(fig)
    x=np.mean(lr,axis=0);y=np.mean(nr,axis=0)
    fig,ax=plt.subplots(figsize=(3.6,3.6));ax.scatter(x,y,s=22,color='#2563eb')
    lo=min(x.min(),y.min())-.5;hi=max(x.max(),y.max())+.7
    ax.plot([lo,hi],[lo,hi],'--',color='#6b7280',lw=.7)
    ax.set(xlim=(lo,hi),ylim=(lo,hi));ax.set_xlabel('Mean linear-head rank',fontsize=9);ax.set_ylabel('Mean NN-head rank',fontsize=9)
    ax.tick_params(labelsize=9);annotations=[]
    for label,xx,yy in zip(short,x,y):
        annotations.append(ax.annotate(label,(xx,yy),xytext=(3,3),textcoords='offset points',fontsize=9,
                                      bbox=dict(facecolor='none',edgecolor='none',pad=0),
                                      arrowprops=dict(arrowstyle='-',lw=.3,color='#888')))
    fig.tight_layout()
    # Resolve crowded labels without changing any plotted point or rank.
    for _ in range(15):
        fig.canvas.draw();renderer=fig.canvas.get_renderer();moved=False
        for i in range(len(annotations)):
            box=annotations[i].get_bbox_patch().get_window_extent(renderer)
            for j in range(i):
                if box.overlaps(annotations[j].get_bbox_patch().get_window_extent(renderer)):
                    a,b=annotations[i].get_position();annotations[i].set_position((a,b+7));moved=True;break
        if not moved:break
    fig.savefig(FIGURES/'appendix_nn_rank.pdf',bbox_inches='tight');plt.close(fig)

def generate(nn):
    n=nn.set_index(['dataset','encoder']); f=fixed.set_index(['dataset','method'])
    old.FIGURES=FIGURES;old.GENERATED=GENERATED;old.nn_metrics=lambda:nn
    appendix_figures(nn)
    plt.style.use('seaborn-v0_8-whitegrid')
    selected=[(d,e,float(f.loc[(d,e),'test_r2']),float(n.loc[(d,e),'test_r2'])) for d in DATASETS[:2] for e in ['resnet50','dinov2_vitb14']]
    fig,ax=plt.subplots(figsize=(9.2,3.25));x=np.arange(4)
    ax.scatter(x-.07,[r[2] for r in selected],s=55,color='#7f7f7f',label='Linear head',zorder=3)
    ax.scatter(x+.07,[r[3] for r in selected],s=55,color='#9467bd',label='NN head',zorder=3)
    for i,(_,_,a,b) in enumerate(selected):
        ax.plot([i-.07,i+.07],[a,b],color='#c7c7c7',lw=1.2)
        ax.text(i-.09,a+.014,f'{a:.3f}',ha='right',fontsize=10)
        ax.text(i+.09,b+.014,f'{b:.3f}',ha='left',fontsize=10)
    ax.set_xticks(x,['Houses\nResNet50','Houses\nDINOv2','Horses\nResNet50','Horses\nDINOv2'])
    ax.set_ylabel('Untouched-test $R^2$');ax.margins(y=.22);ax.legend(frameon=False,loc='upper left',ncol=2)
    ax.spines[['top','right']].set_visible(False);fig.tight_layout()
    fig.savefig(FIGURES/'figure3_nn_robustness.pdf',bbox_inches='tight');plt.close(fig)
    rows=[r'\begin{tabular}{lrrrr}',r'\toprule',r'& \multicolumn{2}{c}{Houses} & \multicolumn{2}{c}{Horses}\\',r'Encoder & Linear & NN & Linear & NN\\',r'\midrule']
    for e,label in ENC.items():
        values=[v for d in DATASETS[:2] for v in [f.loc[(d,e),'test_r2'],n.loc[(d,e),'test_r2']]]
        rows.append(label+' & '+' & '.join(fmt(v) for v in values)+r'\\')
    (TABLES/'generated_nn_motivation.tex').write_text('\n'.join(rows+[r'\bottomrule',r'\end{tabular}'])+'\n')
    rows=[r'\begin{tabular}{lrr}',r'\toprule',r'Encoder & Test MAE & Seed SD\\',r'\midrule']
    for e,label in ENC.items():rows.append(label+' & '+fmt(n.loc[('facial_age',e),'test_mae'],3)+' & '+fmt(n.loc[('facial_age',e),'seed_mae_sd'],3)+r'\\')
    (TABLES/'generated_age_nn.tex').write_text('\n'.join(rows+[r'\bottomrule',r'\end{tabular}'])+'\n')
    gaps={d:float(n.loc[(d,'dinov2_vitb14'),'test_r2']-n.loc[(d,'resnet50'),'test_r2']) for d in DATASETS[:2]}
    lines=['% Generated by generate_neural_assets.py from audited saved predictions.',
           rf'\newcommand{{\HouseNNGap}}{{{gaps[DATASETS[0]]:.3f}}}',rf'\newcommand{{\HorseNNGap}}{{{gaps[DATASETS[1]]:.3f}}}',
           rf'\newcommand{{\NNPackageCommit}}{{\nolinkurl{{{COMMIT}}}}}']
    (GENERATED/'nn_scalars.tex').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':generate(nn_metrics())
