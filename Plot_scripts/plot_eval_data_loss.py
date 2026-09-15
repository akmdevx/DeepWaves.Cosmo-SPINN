

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def moving_average(x, alpha=0.1):
    y = np.zeros_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y
# --------------------------------------------------
# Read logs
# --------------------------------------------------
train1, test1 = np.load('train_losses_DO.npy'),np.load('test_losses_DO.npy')
train2, test2 = np.load('train_losses_PI.npy'),np.load('test_losses_PI.npy')
print(len(train1), len(train2))

# --------------------------------------------------
# Restore first 250 epochs
# --------------------------------------------------
train2 = np.concatenate([train1[:250], train2])
test2  = np.concatenate([test1[:250], test2])

epochs = np.arange(1, 3001)

# --------------------------------------------------
# Plot
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(12,9))

# --------------------------------------------------
# Generalization gap
# --------------------------------------------------

# Generalization gap (Data loss)
ax.fill_between(
    epochs, train1, test1,
    facecolor='C0',
    alpha=0.10,
    hatch='///',
    edgecolor='C0',
    linewidth=0.0,
    zorder=0
)

# Generalization gap (Data + SP)
ax.fill_between(
    epochs, train2, test2,
    facecolor='C1',
    alpha=0.10,
    hatch='\\\\',
    edgecolor='C1',
    linewidth=0.0,
    zorder=0
)

# --------------------------------------------------
# Raw curves (faint)
# --------------------------------------------------
ax.plot(epochs, train1, color='C0', lw=1, alpha=0.18)
ax.plot(epochs, test1, '--', color='C0', lw=1, alpha=0.18)

ax.plot(epochs, train2, color='C1', lw=1, alpha=0.18)
ax.plot(epochs, test2, '--', color='C1', lw=1, alpha=0.18)

# --------------------------------------------------
# Smoothed curves
# --------------------------------------------------
ax.plot(
    epochs,
    moving_average(train1),
    color='C0',
    lw=2.5,
    marker='o',
    markersize=4,
    markevery=200,
    label=r'Data loss ($\mathcal{L}_{\mathrm{data}}$)'
)

ax.plot(
    epochs,
    moving_average(train2),
    color='C1',
    lw=2.5,
    marker='o',
    markersize=4,
    markevery=200,
    label=r'Data + SP loss ($\mathcal{L}_{\mathrm{data}}+\mathcal{L}_{\mathrm{SP}}$)'
)

ax.plot(
    epochs,
    moving_average(test1),
    '--',
    color='C0',
    lw=2.5,
    marker='s',
    markersize=4,
    markevery=200,
)

ax.plot(
    epochs,
    moving_average(test2),
    '--',
    color='C1',
    lw=2.5,
    marker='s',
    markersize=4,
    markevery=200,
)


ax.axvline(
    250,
    color='k',
    linestyle=':',
    linewidth=2,
    alpha=0.8
)


# --------------------------------------------------
# Formatting
# --------------------------------------------------
ax.set_xlabel("Epoch", fontsize=23)
ax.set_ylabel("Data loss", fontsize=23)
ax.set_yscale("log")
ax.tick_params(axis='both', labelsize=23)

# --------------------------------------------------
# Legend 1
# --------------------------------------------------
legend_experiment = ax.legend(
    frameon=False,
    loc="center right",
    title="Training objective",
    fontsize=22,
    title_fontsize=23
)

ax.add_artist(legend_experiment)

# --------------------------------------------------
# Legend 2
# --------------------------------------------------

from matplotlib.patches import Patch

style_handles = [
    Line2D([0],[0], color='k', lw=2.5,
           linestyle='-', marker='o',
           markersize=5, label='Training set'),

    Line2D([0],[0], color='k', lw=2.5,
           linestyle='--', marker='s',
           markersize=5, label='Test set'),

    Line2D([0],[0], color='k',
           linestyle=':',
           lw=2,
           label='SP loss added'),

    Patch(facecolor='C0', alpha=0.10,
          hatch='///',
          edgecolor='C0',
          label='Gap (Data loss)'),

    Patch(facecolor='C1', alpha=0.10,
          hatch='\\\\',
          edgecolor='C1',
          label='Gap (Data + SP)')
]


ax.legend(
    handles=style_handles,
    frameon=False,
    loc="upper right",
    title="Dataset",
    fontsize=22,
    title_fontsize=23
)

plt.title("Data Loss over Epochs", fontsize=23, fontweight="bold")

plt.tight_layout()
plt.savefig("Data_loss_comp.png", dpi=300)
plt.show()
