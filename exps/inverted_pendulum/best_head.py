"""Parse best-head experiments."""

from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd

from exps.plotters import set_figure_params

COLORS = OrderedDict(sim_return="C0", model_return="C2")
LABELS = OrderedDict(sim_return=r"\textbf{H-UCRL}", model_return="Best Head")

set_figure_params(serif=True, fontsize=9)
df = pd.read_pickle("learned_optimistic.pkl")

df["episode"] = 0
EPISODES = {0: 0, 1: 4, 2: 9}
for i in range(9):
    df.iloc[slice(5 * i, 5 * (i + 1)), df.columns.get_loc("episode")] = EPISODES[i % 3]

max_ = (
    df.groupby(["action_cost", "episode", "head"])[["sim_return", "model_return"]]
    .max()
    .max(level=[0, 1])
)

action_costs = [0, 0.1, 0.2]
episodes = [0, 4, 9]
fig, axes = plt.subplots(1, len(action_costs), sharey="row")
fig.set_size_inches(5.5, 2.0)
#
width = 0.8
for i, action_cost in enumerate(action_costs):
    for j, episode in enumerate(episodes):
        for k, type_ in enumerate(["sim_return", "model_return"]):
            axes[i].bar(
                2.5 * j + k,
                max_.loc[(action_cost, episode)][type_],
                width=width,
                color=COLORS[type_],
            )
    axes[i].set_title(f"Action Penalty {action_cost}", fontsize=8)
    axes[i].set_ylim([0, 300])
    axes[i].set_xticks([0.5, 3, 5.5])
    axes[i].tick_params(axis="x", which="major", pad=1)
    axes[i].set_xticklabels([1, 5, 10])

#
for key, label in LABELS.items():
    for ax in axes:
        ax.bar(0, 0, 0, color=COLORS[key], label=label)

handles, labels = axes[-1].get_legend_handles_labels()
axes[-1].legend(
    handles,
    labels,
    loc="upper right",
    frameon=False,
    ncol=2,
    handletextpad=0.3,
    labelspacing=0.1,
    columnspacing=1.0,
    # bbox_to_anchor=(1.0, 1.05)
)

axes[0].set_ylabel("Simulated Return")
axes[1].set_xlabel("Episode Number")
# fig.tight_layout(rect=[0.02, 0.1, 1, 1], pad=0.2)
plt.tight_layout(pad=0.2)
plt.savefig("best_head.pdf")
plt.show()
