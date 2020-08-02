"""Parse num-head experiments."""

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exps.plotters import set_figure_params

COLORS = OrderedDict(sim_return="C0", model_return="C2")
LABELS = OrderedDict(sim_return=r"\textbf{H-UCRL}", model_return="Best Head")

set_figure_params(serif=True, fontsize=9)
df = pd.read_pickle("inverted_mbmpo_num_heads.pkl")
df = df[df.seed != 2]

mean = (
    df.groupby(["action_cost", "model_num_heads", "seed"])
    .best_return.max()
    .mean(level=[0, 1])
)
std = (
    df.groupby(["action_cost", "model_num_heads", "seed"])
    .best_return.max()
    .std(level=[0, 1])
)

action_costs = [0, 0.1, 0.2]
num_heads = [5, 10, 20, 50, 100]
fig, axes = plt.subplots(1, len(action_costs), sharey="row")
fig.set_size_inches(5.5, 2.0)
#
width = 0.8
for i, action_cost in enumerate(action_costs):
    for j, num_head in enumerate(num_heads):
        axes[i].bar(
            j,
            mean.loc[(action_cost, num_head)],
            width=width,
            yerr=max(10, std.loc[(action_cost, num_head)]),
            color="C2",
        )
    axes[i].set_title(f"Action Penalty {action_cost}", fontsize=8)
    axes[i].set_ylim([0, 300])
    axes[i].set_xticks(np.arange(5))
    axes[i].tick_params(axis="x", which="major", pad=1)
    axes[i].set_xticklabels(num_heads)


axes[0].set_ylabel("Episode Return Return")
axes[1].set_xlabel("Number of Heads")
# fig.tight_layout(rect=[0.02, 0.1, 1, 1], pad=0.2)
plt.tight_layout(pad=0.2)
plt.savefig("num_head.pdf")
plt.show()
