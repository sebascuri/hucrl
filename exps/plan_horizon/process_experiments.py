"""Python Script Template."""
import matplotlib.pyplot as plt
import numpy as np

from exps.post_process import parse_results

base_dir = "runs/Cartpoleenv"
df = parse_results(base_dir, "MBMPO")
plan_horizon = df.plan_horizon.unique()

fig, axes = plt.subplots(1, 2, sharey="row")
for h in plan_horizon:
    mean = df[df.plan_horizon == h].best_return.mean(level=0)
    std = df[df.plan_horizon == h].best_return.std(level=0).fillna(0)
    axes[0].plot(mean, label=f"Plan Horizon {h}")
    axes[0].fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.3)

for h in plan_horizon:
    mean = df[df.plan_horizon == h].best_model_return.mean(level=0)
    std = df[df.plan_horizon == h].best_model_return.std(level=0).fillna(0)
    axes[1].plot(mean, label=f"Plan Horizon {h}")
    axes[1].fill_between(np.arange(len(mean)), mean - std, mean + std, alpha=0.3)


axes[0].set_ylabel("Returns")
axes[0].set_xlabel("Num Episode")
axes[1].set_xlabel("Num Episode")

axes[0].set_title("Environment Return")
axes[1].set_title("Model Return")

axes[1].legend(loc="best")

fig.tight_layout()
plt.show()
