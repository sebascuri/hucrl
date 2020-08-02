"""Parse simulation returns experiments."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exps.plotters import COLORS, LABELS, set_figure_params

set_figure_params(serif=True, fontsize=9)
df = pd.read_pickle("inverted_mbmpo_scale.pkl")
df = df[df.seed < 4]

action_costs = [0, 0.1, 0.2]
fig, axes = plt.subplots(1, len(action_costs), sharey="row")
# axes = [axes]
fig.set_size_inches(5.5, 2.0)

# action_costs = df.action_cost.unique()
explorations = df.exploration.unique()
model_kinds = ["ProbabilisticEnsemble"]  # , 'DeterministicEnsemble']
for i, action_cost in enumerate(action_costs):
    for model_kind in model_kinds:
        learn_df = df[(df.action_cost == action_cost) & (df.model_kind == model_kind)]

        mean = (
            learn_df.groupby(["exploration", "episode", "seed"])
            .sim_return.max()
            .mean(level=[0, 1])
        )
        std = (
            learn_df.groupby(["exploration", "episode", "seed"])
            .sim_return.max()
            .std(level=[0, 1])
            .fillna(20)
        )

        episodes = np.arange(len(mean.expected))
        for exploration in explorations:
            axes[i].plot(episodes, mean[exploration], color=COLORS[exploration])
            axes[i].fill_between(
                episodes,
                mean[exploration] - std[exploration],
                mean[exploration] + std[exploration],
                alpha=0.2,
                color=COLORS[exploration],
            )
    # axes[i].axhline(OPTIMAL[action_cost], linestyle='--', color='grey')
    axes[i].set_ylim([-100, 400])
    axes[i].set_xlabel("Episode")
    axes[i].set_title(f"Action Penalty {action_cost}")
    axes[i].set_xlim([0, 20])

axes[0].set_ylabel("Simulated Return")
for key, label in LABELS.items():
    axes[0].plot(0, 0, color=COLORS[key], linestyle="-", label=label)

handles, labels = axes[0].get_legend_handles_labels()

axes[0].legend(
    handles,
    labels,
    loc="lower right",
    frameon=False,
    handletextpad=0.3,
    labelspacing=0.1,
    columnspacing=1.0,
    # bbox_to_anchor=(1.04, -.08)
)

# axes[2].legend(loc='upper left', frameon=False, ncol=1)
# plt.show()
plt.tight_layout(pad=0.2)
# plt.show()
plt.savefig("simulated_returns.pdf")
