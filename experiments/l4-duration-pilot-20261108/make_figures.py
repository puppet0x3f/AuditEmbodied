from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
other = root.parent / "l4-duration-relevel-20261108"
reports = [json.loads((p / "report.json").read_text()) for p in (root, other)]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
for ax, report, title in zip(axes, reports, ("All local phases retimed (24 states)", "Only relevel retimed (10 states)")):
    values = [report["by_scale"][str(s)] for s in (.75, 1., 1.5)]
    x = np.arange(3)
    n = values[0]["episodes"]
    for offset, key, label, color in ((-.18, "keeper_continuous", "Keeper preserved", "#3b82a0"),
                                     (.18, "safe_successes", "Safe task completion", "#e39b40")):
        counts = [v[key] for v in values]
        bars = ax.bar(x+offset, np.array(counts)/n, .34, label=label, color=color)
        ax.bar_label(bars, labels=[f"{c}/{n}" for c in counts], padding=3, fontsize=9)
    ax.set(xticks=x, xticklabels=["0.75x", "1.00x", "1.50x"], ylim=(0, 1.17),
           ylabel="Fraction of paired fault states", xlabel="Local time-scale intervention", title=title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", fontsize=8)
fig.suptitle("Recovery timing changes support safety and completion differently", fontsize=13)
fig.savefig(root / "duration-mechanism.png", dpi=200)
fig.savefig(root / "duration-mechanism.pdf")
plt.close(fig)

rows = []
with (other / "pilot-trials.jsonl").open() as stream:
    rows = [json.loads(line) for line in stream]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
for ax, scenario in zip(axes, ("left_loss", "right_loss")):
    for scale, color in zip((.75, 1., 1.5), ("#d55e00", "#6b7280", "#009e73")):
        r = next(r for r in rows if r["scenario"] == scenario and r["candidate"] != "R_global" and r["time_scale"] == scale)
        keeper = 1 if scenario == "left_loss" else 0
        frames = [f for f in r["trace"] if f["phase"] == "recovery"]
        t = [f["time"]-r["prefix_state"]["time"] for f in frames]
        y = [int(f["evaluator"]["holds"][keeper]) for f in frames]
        ax.step([0]+t, [1]+y, where="post", color=color, linewidth=1.7,
                linestyle="--" if scale == 1 else "-", label=f"{scale:.2f}x, {r['protection']['keeper_grasp_lost_steps']} interrupted steps")
    ax.set(title=f"Seed 5: {scenario.replace('_', ' ')}", xlabel="Seconds after the identical fault prefix",
           ylabel="Keeper grasp present", ylim=(-.15, 1.2), yticks=[0, 1])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", fontsize=8)
fig.suptitle("Phase isolation: approach and closing actions are identical", fontsize=13)
fig.savefig(root / "relevel-support-traces.png", dpi=200)
fig.savefig(root / "relevel-support-traces.pdf")
