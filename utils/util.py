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

def lap_fft(f: torch.Tensor, Lx=1000.0):
    """
    Batched FFT Laplacian.
    f: (..., nx, nx, nx) real tensor
       supports batch dimensions (B, nx, nx, nx) or (nx, nx, nx)
    returns same shape as f
    """

    device = f.device
    dtype = f.dtype

    # ---------------------------------
    # Get spatial dimension
    # ---------------------------------
    nx = f.shape[-1]
    dx = Lx / nx

    # ---------------------------------
    # Build Fourier wave numbers (broadcastable)
    # ---------------------------------
    k_lin_np = 2.0 * np.pi / Lx * np.arange(-nx/2, nx/2)
    k_lin = torch.tensor(k_lin_np, dtype=dtype, device=device)

    kx, ky, kz = torch.meshgrid(k_lin, k_lin, k_lin, indexing="ij")

    # apply the same shift as NumPy
    kx = torch.fft.ifftshift(kx)
    ky = torch.fft.ifftshift(ky)
    kz = torch.fft.ifftshift(kz)

    k_sq = kx**2 + ky**2 + kz**2       # shape: (nx, nx, nx)

    # make k_sq broadcastable over batches
    while k_sq.dim() < f.dim():
        k_sq = k_sq.unsqueeze(0)

    # ---------------------------------
    # FFT (batched automatically)
    # ---------------------------------
    F_hat = torch.fft.fftn(f, dim=(-3, -2, -1))

    # ---------------------------------
    # Multiply by -|k|^2
    # ---------------------------------
    lap_hat = -k_sq * F_hat

    # ---------------------------------
    # Inverse FFT
    # ---------------------------------
    lap = torch.fft.ifftn(lap_hat, dim=(-3, -2, -1)).real

    return lap


def compute_dt_batched(a, da, H0, omega_matter, omega_lambda, n_quad=1000, eps=1e-12):
    """
    a, da: shape (B,1)
    Returns dt: shape (B,1)
    """

    device = a.device
    dtype  = a.dtype

    B = a.shape[0]

    # Quadrature points (n_quad,)
    dx = 1.0 / n_quad
    lin = torch.linspace(0.5 * dx, 1.0 - 0.5 * dx, n_quad,
                         device=device, dtype=dtype)       # (n_quad,)
    lin = lin.unsqueeze(0)                                 # (1, n_quad)

    # Expand a, a+da to (B, n_quad)
    a_exp  = a * lin                                       # (B, n_quad)
    a2_exp = (a + da) * lin                                # (B, n_quad)

    # Compute adot(a)
    adot  = H0 * torch.sqrt(torch.clamp(
            omega_matter / (a_exp + eps) +
            omega_lambda * a_exp**2,
            min=0.0))

    t = torch.mean(1.0 / adot, dim=1, keepdim=True) * a    # (B,1)

    # Compute adot(a+da)
    adot2 = H0 * torch.sqrt(torch.clamp(
            omega_matter / (a2_exp + eps) +
            omega_lambda * a2_exp**2,
            min=0.0))

    t2 = torch.mean(1.0 / adot2, dim=1, keepdim=True) * (a + da)  # (B,1)

    # Δt = t(a+da) – t(a)
    dt = t2 - t                                        # (B,1)

    # Supercomoving time factor
    dt = dt * a.pow(-2)                                # (B,1)

    return dt

def compute_pres(gen, g2, labels, labels2, stats,dx=1000.0/64.0):
    """
    Arguments:
      - gen: (B, 3, Nx, Ny, Nz) tensor: generator output at time t1 (Re, Im, phi maybe at channel 2)
      - g2:  (B, 3, Nx, Ny, Nz) tensor: generator output at time t2 (Re, Im, phi)
      - labels, labels2: tensors used to compute scale factors a and a2
      - stats: dict with "mu","sigma" used in inverse_global (pass the same stats you had)
    """

    device = gen.device
    dtype = gen.dtype


    #a_min_l =-0.8076007221764763 #to -0.8048255392272029# -0.8273942837611394
    a_min_l = -0.8273942837611394
    a_max_l = -0.8048255392272029

    # build scale factors a, a2 (keep them 1D arrays for compute_dt)
    labels_proc = (labels * 0.5 + 0.5)
    labels_proc = labels_proc * (a_max_l - a_min_l)+a_min_l
    a = 10 ** labels_proc  # shape (B,)

    labels2_proc = (labels2 * 0.5 + 0.5)
    labels2_proc = labels2_proc * (a_max_l - a_min_l)+a_min_l
    a2 = 10 ** labels2_proc  # shape (B,)

    # --- cosmology constants used in compute_dt and a_dott ---
    H0_local = 0.1
    O_m0 = omega_matter
    O_l0 = omega_lambda

    #print(a.shape)
    denom = compute_dt_batched(a, a2-a, H0 = H0_local, omega_matter = O_m0, omega_lambda = O_l0)
    # compute denom (dt) using numpy, then convert to torch properly

    # reshape a,a2 for broadcasting to spatial dims: (B,1,1,1)
    a = torch.tensor(a, device=device, dtype=dtype).view(-1, 1, 1, 1)
    a2 = torch.tensor(a2, device=device, dtype=dtype).view(-1, 1, 1, 1)
    #denom = a2 - a
   
    # Rescale from [-1, 1] to [0, 1]
    gen = 0.5 * gen + 0.5
    g2 = 0.5 * g2 + 0.5
    #print(gen[:,0,:,:,:])
    # Inverse normalization (expect stats provided)
    def inverse_global(y, stats):
        mu, sigma = stats["mu"], stats["sigma"]
        return y * sigma + mu

    if stats is None:
        raise ValueError("stats must be passed to compute_pres (dict with 'mu' and 'sigma')")
    gen = inverse_global(gen, stats)
    g2 = inverse_global(g2, stats)

    # Inverse arcsinh
    gen = torch.sinh(gen)
    g2 = torch.sinh(g2)

    # cosmological a_dot
    a_dott = a * H0_local * torch.sqrt(O_m0 * a**(-3) + O_l0)  # shape (B,1,1,1)

    # kinetic / potential prefactors
    da = 1e-5
    kin_fac = 1.0
    pot_fac = a
    #print(denom)
    # split real & imag channels (expected gen/g2 shapes: (B, C, Nx, Ny, Nz))
    r1 = gen[:, 0, :, :, :]
    i1 = gen[:, 1, :, :, :]
    r2 = g2[:, 0, :, :, :]
    i2 = g2[:, 1, :, :, :]

    # compute densities (|psi|^2)
    rho1 = r1**2 + i1**2
    rho2 = r2**2 + i2**2

    #if gen.shape[1] > 2:
    phi1 = gen[:, 2, :, :, :]  # (B,Nx,Ny,Nz)

    lap_r1 = lap_fft(r1)#laplacian_3d_torch(r1, dx=dx)
    lap_i1 = lap_fft(i1)#laplacian_3d_torch(i1, dx=dx)
    lap_V = lap_fft(phi1)#laplacian_3d_torch(phi1, dx =dx)
    denom = torch.tensor(denom, device=device, dtype=dtype).view(-1, 1, 1, 1)
    #print(denom.shape)
    # --- Build the PDE residuals
    lhs_re = (r2 - r1) / denom# *a_dott    # broadcasting now safe: denom is (B,1,1,1)
    lhs_im = (i2 - i1) / denom# *a_dott

    PREF_A = (1.0 / m_per_hbar)
    PREF_B = (m_per_hbar)

    # RHS terms (note lap_i1 etc are torch tensors now)
    res_gen_re = PREF_A * kin_fac * lap_i1 - pot_fac * PREF_B * phi1 * i1
    res_gen_im = -PREF_A * kin_fac * lap_r1 + pot_fac * PREF_B * phi1 * r1

    # full residuals: lhs + rhs
    res_re = lhs_re + res_gen_re
    res_im = lhs_im + res_gen_im
    res_V = lap_V - 4 * torch.pi * G1 * (rho1 - rho_bar)

    res_sq = res_re**2 + res_im**2 + res_V**2  # (B,Nx,Ny,Nz)

    per_sample_loss = res_sq.view(res_sq.shape[0], -1).mean(dim=1)  # (B,)
    return per_sample_loss.mean()


def get_potential_batched(rho):
    """
    rho: (B, Nx, Ny, Nz)
    returns V: same shape
    """
    device = rho.device
    dtype = rho.dtype
    Lx = 1000.0
    # ---------------------------------
    # Get spatial dimension
    # ---------------------------------
    nx = rho.shape[-1]
    dx = Lx / nx

    # ---------------------------------
    # Build Fourier wave numbers (broadcastable)
    # ---------------------------------
    k_lin_np = 2.0 * np.pi / Lx * np.arange(-nx/2, nx/2)
    k_lin = torch.tensor(k_lin_np, dtype=dtype, device=device)

    kx, ky, kz = torch.meshgrid(k_lin, k_lin, k_lin, indexing="ij")

    # apply the same shift as NumPy
    kx = torch.fft.ifftshift(kx)
    ky = torch.fft.ifftshift(ky)
    kz = torch.fft.ifftshift(kz)

    k_sq = kx**2 + ky**2 + kz**2       # shape: (nx, nx, nx)

    # subtract mean per batch
    rho = rho - rho.mean(dim=(1,2,3), keepdim=True)

    rho_k = torch.fft.fftn(4.0 * torch.pi * G1 * rho, dim=(-3,-2,-1))

    denom = k_sq.clone()
    denom = denom.unsqueeze(0)  # (1, Nx, Ny, Nz)
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)

    V_hat = -rho_k / denom
    V = torch.fft.ifftn(V_hat, dim=(-3,-2,-1)).real

    return V

def Laplacefft_batched(F):
    """
    F: (B, Nx, Ny, Nz)
    """
    device = F.device
    dtype =F.dtype
    Lx = 1000.0
    # ---------------------------------
    # Get spatial dimension
    # ---------------------------------
    nx = F.shape[-1]
    dx = Lx / nx

    # ---------------------------------
    # Build Fourier wave numbers (broadcastable)
    # ---------------------------------
    k_lin_np = 2.0 * np.pi / Lx * np.arange(-nx/2, nx/2)
    k_lin = torch.tensor(k_lin_np, dtype=dtype, device=device)

    kx, ky, kz = torch.meshgrid(k_lin, k_lin, k_lin, indexing="ij")

    # apply the same shift as NumPy
    kx = torch.fft.ifftshift(kx)
    ky = torch.fft.ifftshift(ky)
    kz = torch.fft.ifftshift(kz)

    k_sq = kx**2 + ky**2 + kz**2       # shape: (nx, nx, nx)
    F_hat = torch.fft.fftn(F, dim=(-3,-2,-1))
    lap_hat = -k_sq.unsqueeze(0) * F_hat
    return torch.fft.ifftn(lap_hat, dim=(-3,-2,-1)).real


def drift_kick_1storder_batched(psi, kin_fac, pot_fac,dt):
    """
    psi: (B, Nx, Ny, Nz) complex
    """

    # ---- (1/2) kick ----
    rho_tot = torch.abs(psi)**2
    V = get_potential_batched(rho_tot))

    psi_kick = (
        1 - 1j * m/hbar * dt * pot_fac / 2.0 * V
    ) * psi

    # ---- drift ----
    psi_lap = (
        Laplacefft_batched(psi_kick.real)
        + 1j * Laplacefft_batched(psi_kick.imag)
    )

    psi_drift = psi_kick + dt * kin_fac * (
        -1j * hbar/m / 2.0
    ) * psi_lap

    # ---- (1/2) kick ----
    rho_tot = torch.abs(psi_drift)**2
    V = get_potential_batched(rho_tot)

    psi_out = (
        1 - 1j * m/hbar * dt * pot_fac / 2.0 * V
    ) * psi_drift

    return psi_out

def compute_pres_KDK(gen, g2, labels, labels2, stats,dx=1000.0/64.0):
    """
    Arguments:
      - gen: (B, 2, Nx, Ny, Nz) tensor: generator output at time t1 (Re, Im, phi maybe at channel 2)
      - g2:  (B, 2, Nx, Ny, Nz) tensor: generator output at time t2 (Re, Im)
      - labels, labels2: tensors used to compute scale factors a and a2
      - stats: dict with "mu","sigma" used in inverse_global (pass the same stats you had)
    """

    device = gen.device
    dtype = gen.dtype
    #Usef for SR task
    a_min_l = -1.0948410478153783
    a_max_l =  -1.0945714257315697
    # build scale factors a, a2 (keep them 1D arrays for compute_dt)
    labels_proc = (labels * 0.5 + 0.5)
    labels_proc = labels_proc * (a_max_l - a_min_l)+a_min_l
    a = 10 ** labels_proc  # shape (B,)

    labels2_proc = (labels2 * 0.5 + 0.5)
    labels2_proc = labels2_proc * (a_max_l - a_min_l)+a_min_l
    a2 = 10 ** labels2_proc  # shape (B,)

    # --- cosmology constants used in compute_dt and a_dott ---
    H0_local = 0.1
    O_m0 = omega_matter
    O_l0 = omega_lambda

    #print(a.shape)
    dt_batched = compute_dt_batched(a, a2-a, H0 = H0_local, omega_matter = O_m0, omega_lambda = O_l0)
    # reshape a,a2 for broadcasting to spatial dims: (B,1,1,1)
    a = torch.tensor(a, device=device, dtype=dtype).view(-1, 1, 1, 1)
    a2 = torch.tensor(a2, device=device, dtype=dtype).view(-1, 1, 1, 1)
    #denom = a2 - a
   
    # Rescale from [-1, 1] to [0, 1]
    gen = 0.5 * gen + 0.5
    g2 = 0.5 * g2 + 0.5
    #print(gen[:,0,:,:,:])
    # Inverse normalization (expect stats provided)
    def inverse_global(y, stats):
        mu, sigma = stats["mu"], stats["sigma"]
        return y * sigma + mu

    if stats is None:
        raise ValueError("stats must be passed to compute_pres (dict with 'mu' and 'sigma')")
    gen = inverse_global(gen, stats)
    g2 = inverse_global(g2, stats)

    # Inverse arcsinh
    gen = torch.sinh(gen)
    g2 = torch.sinh(g2)

    # cosmological a_dot
    a_dott = a * H0_local * torch.sqrt(O_m0 * a**(-3) + O_l0)  # shape (B,1,1,1)

    # kinetic / potential prefactors
    da = 1e-6

    # =========================
    # factors
    # =========================
    kin_fac = 0.5 * (a**-2 + (a + da)**-2)
    pot_fac = 0.5 * (a**-1 + (a + da)**-1)
    #print(denom)
    # split real & imag channels (expected gen/g2 shapes: (B, C, Nx, Ny, Nz))
    r1 = gen[:, 0, :, :, :]
    i1 = gen[:, 1, :, :, :]
    r2 = g2[:, 0, :, :, :]
    i2 = g2[:, 1, :, :, :]

    # compute densities (|psi|^2)
    rho1 = r1**2 + i1**2
    rho2 = r2**2 + i2**2
    
    psi1 = r1 +1j*i1
    psi2 = r2 + 1j*i2
    dt_batched       = dt_batched.view(-1, 1, 1, 1)
    kin_fac  = kin_fac.view(-1, 1, 1, 1)
    pot_fac  = pot_fac.view(-1, 1, 1, 1)
    Psi_1st = drift_kick_1storder_batched(psi1,kin_fac,pot_fac,dt_batched) 
    #print(Psi_1st)
    res = torch.mean(torch.abs(psi2 - Psi_1st)**2)

    return res.mean()
