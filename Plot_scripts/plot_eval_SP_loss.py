import matplotlib.pyplot as plt

# =========================================================
# RUN 1 (epochs 21-80)
# =========================================================
epochs1 =  list(range(250, 3001,50))


# =========================================================
# RUN 2 (epochs 1-60)
# =========================================================
epochs2 = list(range(250, 3001,50))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def ema(x, alpha=0.1):
    y = np.zeros_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y



train1 = np.load('./Loss_data/SP_loss_DO.npy')
train2 = np.load('./Loss_data/SP_loss_PI.npy')


# --------------------------------------------------
# Smooth
# --------------------------------------------------
train1_s = ema(train1, alpha=0.50)
train2_s = ema(train2, alpha=0.50)


# --------------------------------------------------
# Plot
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 9))


#print(len(train2))

# Raw curves (faint)
ax.plot(epochs1, train1, color='C0', alpha=0.30)
ax.plot(epochs2, train2, color='C1', alpha=0.30)


ax.plot(
    epochs1, train1_s,
    color='C0', lw=3,
    label=r'Data loss ($\mathcal{L}_{\mathrm{data}}$)'
)

ax.plot(
    epochs2, train2_s,
    color='C1', lw=3,
    label=r'Data + SP loss ($\mathcal{L}_{\mathrm{data}} + \mathcal{L}_{\mathrm{SP}}$)'
)


# --------------------------------------------------
# Labels
# --------------------------------------------------
ax.set_xlabel("Epoch", fontsize=23)
ax.set_ylabel("Normalized SP loss", fontsize=23)
ax.tick_params(axis='both', labelsize=23)


legend_experiment = ax.legend(
    frameon=False,
    loc="lower left",
    title="Training objective",fontsize=22,       # legend entries
    title_fontsize=23  # legend title
)

ax.add_artist(legend_experiment)


plt.title('SP Loss over Epochs', fontsize=23, fontweight='bold')
plt.tight_layout()
plt.show()
