import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler



##################
# Global Constants
##################
L = 1000.0
m_22 = 2.5
G1 = 4.30241002e-6
hbar = 1.71818134e-87
ev_to_msun = 8.96215334e-67
ev_to_internal = 8.05478173e-56
c = 299792.458
m = m_22 * 1.0e-22 * ev_to_msun
m_per_hbar = m / hbar

h = 1.0
H0 = 0.1 * h
rho_crit = 3.0 * H0**2 / (8.0 * torch.pi * G1)
omega_matter = 0.27
omega_lambda = 0.73
rho_bar = omega_matter * rho_crit
