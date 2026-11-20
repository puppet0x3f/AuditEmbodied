"""A task illustration from actual, observation-verified replay pixels."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
plt.rcParams.update({'font.size':7,'pdf.fonttype':42})
reports=json.loads((HERE/'waiting-replay-report.json').read_text())
fig,axes=plt.subplots(2,5,figsize=(7.05,2.75))
titles=('Common fault','After 1 s waiting','Table placement','First pickup close','Final verification')
for i,report in enumerate(reports):
    for j,ax in enumerate(axes[i]):
        rgb=plt.imread(HERE/'waiting-keyframes'/f"{report['hold']}-{j}.png")
        ax.imshow(rgb[165:512,0:512]);ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_color('#aab2b8');spine.set_linewidth(.4)
        if i==0:ax.set_title(titles[j],fontsize=7,pad=3)
        f=report['frames'][j]
        ax.set_xlabel(f"{f['seconds']:.2f} s; grasp {sum(f['holds'])}/2",fontsize=6.3,labelpad=2)
        if j==0:ax.set_ylabel('Registered' if i==0 else 'Position only',fontsize=7)
fig.subplots_adjust(left=.055,right=.997,bottom=.09,top=.92,wspace=.04,hspace=.25)
for ext in ('pdf','png'):
    p=HERE/f'waiting-execution.{ext}';fig.savefig(p,dpi=220,bbox_inches='tight')
    (ROOT/'paper/figures'/p.name).write_bytes(p.read_bytes())
