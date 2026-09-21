"""Render fixed saved-development comparisons from aggregate results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
r = Path(__file__).resolve().parent
x = json.loads((r/'paired-development.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained')
colors = ['#727272', '#0072B2', '#D55E00', '#009E73']
for c, label in enumerate(['Same figure','Same paper','Related papers','Unrelated papers']):
    axes[0].plot([2,4,8],[x['models'][f'public{k}k']['f1'][c] for k in [2,4,8]], marker='o',label=label,color=colors[c])
axes[0].set(xlabel='Public adaptation updates (thousands)',ylabel='CXI development class F1',xticks=[2,4,8],ylim=(.45,1),title='(a) Class tradeoffs across selected endpoints')
axes[0].legend(fontsize=8,loc='lower right')
keys=['public2k -> public4k','public4k -> public8k','answer_sft -> gspo512','public4k -> gspo512']
labels=['2k → 4k','4k → 8k','Answer SFT → GSPO512','Classifier → GSPO512']
y=np.arange(len(keys))
for i,k in enumerate(keys):
    z=x['comparisons'][k]
    axes[1].barh(i,-z['broken'],color='#D55E00',label='Correct → wrong' if i==0 else None)
    axes[1].barh(i,z['corrected'],color='#0072B2',label='Wrong → correct' if i==0 else None)
    axes[1].text(-z['broken']-2,i,str(z['broken']),ha='right',va='center',fontsize=9)
    axes[1].text(z['corrected']+2,i,str(z['corrected']),va='center',fontsize=9)
axes[1].set(yticks=y,yticklabels=labels,xlabel='Examples changed (794 fixed DEV pairs)',title='(b) Corrections and regressions',xlim=(-70,100))
axes[1].invert_yaxis();axes[1].axvline(0,color='#555',lw=.7);axes[1].legend(fontsize=8,loc='lower right')
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
fig.savefig(r/'paired-development.pdf');fig.savefig(r/'paired-development.png',dpi=160)
