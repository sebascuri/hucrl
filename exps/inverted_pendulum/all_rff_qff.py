"""Parse RFF/QFF experiments."""
import matplotlib.pyplot as plt
import pandas as pd

from exps.plotters import COLORS, LABELS, set_figure_params

set_figure_params(serif=True, fontsize=9)
df = pd.read_pickle("inverted_mbmpo_rff.pkl")
dfqff = df[
    (df.action_cost == 0)
    & df.seed.isin([2, 3])
    & (df.model_feature_approximation == "QFF")
]
dfrff = df[
    (df.action_cost == 0)
    & ~df.seed.isin([2, 3])
    & (df.model_feature_approximation == "RFF")
]
df1 = df[(df.action_cost > 0)]
thompson_rff = pd.concat((dfqff, dfrff, df1))
qff = pd.read_pickle("inverted_mbmpo_qff_optimistic_expected.pkl")
rff = pd.read_pickle("inverted_mbmpo_rff_optimistic_expected.pkl")

df = pd.concat((thompson_rff, qff, rff), sort=True)

action_costs = [0, 0.1, 0.2]
fig, axes = plt.subplots(1, len(action_costs), sharey="row")
fig.set_size_inches(5.5, 2.0)

# action_costs = df.action_cost.unique()
model_kinds = ["RFF", "QFF"]  # , 'DeterministicEnsemble']

mean = (
    df.groupby(["action_cost", "model_feature_approximation", "exploration", "seed"])
    .best_return.max()
    .mean(level=[0, 1, 2])
)
std = (
    df.groupby(["action_cost", "model_feature_approximation", "exploration", "seed"])
    .best_return.max()
    .std(level=[0, 1, 2])
    .fillna(0)
)

action_costs = mean.index.unique(level=0)
features = mean.index.unique(level=1)
explorations = mean.index.unique(level=2)

width = 0.8
for i, action_cost in enumerate(action_costs):
    for j, feature in enumerate(features):
        for k, exploration in enumerate(explorations):
            axes[i].bar(
                (1.7 + len(features)) * j + k,
                mean.loc[(action_cost, feature, exploration)],
                width=width,
                yerr=max(10, std.loc[(action_cost, feature, exploration)]),
                color=COLORS[exploration],
            )
    axes[i].set_title(f"Action Penalty {action_cost}")
    axes[i].set_ylim([0, 350])
    axes[i].set_xticks([1, 4.6])
    axes[i].set_xticklabels(features)
    # axes[i].axhline(OPTIMAL[action_cost], linestyle='--', color='grey')

for key, label in LABELS.items():
    axes[1].bar(0, 0, 0, color=COLORS[key], label=label)

handles, labels = axes[1].get_legend_handles_labels()

axes[0].set_ylabel("Episode Return")
# axes[1].set_xlabel('Number of Features')

axes[1].legend(
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

# plt.tight_layout(pad=0.2)

# plt.show()
# for i, action_cost in enumerate(action_costs):
#     for model_kind in model_kinds:
#         learn_df = df[(df.action_cost == action_cost) & (df.model_kind == model_kind)]
#
#         mean = learn_df.groupby(
#             ['exploration', 'episode', 'seed']).best_return.min().mean(
#             level=[0, 1])
#         std = learn_df.groupby(
#             ['exploration', 'episode', 'seed']).best_return.min().std(
#             level=[0, 1]).fillna(20)
#
#         episodes = np.arange(len(mean.expected))
#         for exploration in explorations:
#             axes[i].plot(episodes, mean[exploration], color=COLORS[exploration])
#             axes[i].fill_between(episodes,
#                                  mean[exploration] - std[exploration],
#                                  mean[exploration] + std[exploration],
#                                  alpha=0.2, color=COLORS[exploration])
#     # axes[i].axhline(OPTIMAL[action_cost], linestyle='--', color='grey')
#     axes[i].set_ylim([0, 300])
#     axes[i].set_xlabel('Episode')
#     axes[i].set_title(f"Action Penalty {action_cost}")
#     axes[i].set_xlim([0, 20])
#
# axes[0].set_ylabel('Uncertainty Scale')
# for key, label in LABELS.items():
#     axes[0].plot(0, 0, color=COLORS[key], linestyle='-', label=label)
#
# handles, labels = axes[0].get_legend_handles_labels()
#
# axes[0].legend(handles, labels, loc='lower right', frameon=False,
#                handletextpad=0.3, labelspacing=0.1, columnspacing=1.,
#                # bbox_to_anchor=(1.04, -.08)
#                )
#
# # axes[2].legend(loc='upper left', frameon=False, ncol=1)
# # plt.show()
fig.tight_layout(pad=0.2)
plt.savefig("all_rff.pdf")

# plt.show()
