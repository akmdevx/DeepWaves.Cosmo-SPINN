import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import glob
import os
import re
import math
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from ./Models import SR_Net.py

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

##################
# Global Constants
##################
L = 1000.0 # in kpc
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



from numpy.random import randint


def power_spectrum_3d(delta, Lbox):
    """
    Computes isotropic 3D power spectrum P(k).

    Parameters
    ----------
    delta : 3D array of shape (N, N, N)
        Density contrast field delta(x) = (rho - rho_mean) / rho_mean
    Lbox  : float
        Box size in Mpc/h

    Returns
    -------
    k  : array
        Wavenumber in h/Mpc
    Pk : array
        Power spectrum in (Mpc/h)^3
    """
    N = delta.shape[0]
    V = Lbox**3   # (Mpc/h)^3

    # FFT and power — normalization gives P(k) in (Mpc/h)^3
    delta_k = np.fft.fftn(delta)
    pk3d    = (np.abs(delta_k)**2) * V / N**6

    # Physical k grid in h/Mpc
    kfreq = np.fft.fftfreq(N, d=Lbox/N) * 2 * np.pi  # h/Mpc
    kx, ky, kz = np.meshgrid(kfreq, kfreq, kfreq, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2).flatten()
    pk3d  = pk3d.flatten()

    # Remove k=0 (DC component)
    nonzero = k_mag > 0
    k_mag = k_mag[nonzero]
    pk3d  = pk3d[nonzero]

    # Log-spaced bins from fundamental mode to Nyquist
    k_fund    = 2 * np.pi / Lbox              # h/Mpc — largest scale in box
    k_nyquist = np.pi * N / Lbox              # h/Mpc — smallest scale (Nyquist)
    k_bins    = np.logspace(np.log10(k_fund), np.log10(k_nyquist), N // 2)

    # Fast vectorized binning
    counts, _ = np.histogram(k_mag, bins=k_bins)
    pk_sum, _ = np.histogram(k_mag, bins=k_bins, weights=pk3d)
    k_sum,  _ = np.histogram(k_mag, bins=k_bins, weights=k_mag)

    # Only keep bins that have modes
    mask   = counts > 0
    k_vals = k_sum[mask]  / counts[mask]   # mean k in bin  [h/Mpc]
    Pk     = pk_sum[mask] / counts[mask]   # mean P(k)      [(Mpc/h)^3]

    return k_vals, Pk


# Your parameters
m22     = 2.5          # m in units of 1e-22 eV
Omega_m = 0.27
H0      = 100        # km/s/Mpc
h       = H0 / 100     

def k_jeans_paper(a, m22, Omega_m=0.27, H0=100):
    """
    Comoving Jeans wavenumber in 1/Mpc (not h/Mpc!)
    Eq. 15 from paper
    """
    return (44.7 * (6/a)**0.25 
                 * (Omega_m/0.3)**0.25 
                 * (H0/70)**0.5 
                 * m22**0.5)        # 1/Mpc

def k_jeans_hMpc(a, m22, Omega_m=0.27, H0=100):
    """Convert to h/Mpc by dividing by h"""
    h = H0/100
    return k_jeans_paper(a, m22, Omega_m, H0) / h   # h/Mpc


def T_fdm_approx(k, a, m22, Omega_m_h2):
    """
    Approximate FDM transfer function relative to CDM
    T(k,a) = P_FDM / P_CDM at scale factor a
    Uses fitting formula from Hu et al. 2000
    """
    kJ = k_jeans_hMpc(a, m22, Omega_m_h2)
    print(kJ)
    x  = 1.61 * m22**(1/18) * (k / kJ)
    return np.cos(x**3) / (1 + x**8),kJ   # approximate
#k_arr = np.logspace(np.log10(6.28), np.log10(400), 200)
#T     = T_fdm_approx(k, a, m22, Omega_m_h2)



from scipy.ndimage import gaussian_filter



##################
# Visualization
##################

def density_to_contrast(rho):
    mean = np.mean(rho)
    return (rho - mean) / mean

import os

from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
import os

from matplotlib.gridspec import GridSpecFromSubplotSpec

##################
# Visualization
##################
def sample_images3(epoch, G_model, X_train, labels, ics,ics_real, stats,max_val, min_val, save_dir="./superres_reals_res_pi"):
    os.makedirs(save_dir, exist_ok=True)



    r, c = 4, 6
    samples_per_row = 4
    n_examples = 4
    
    # ------------------------------------------------------------
    # Get the physical scale factor for every sample
    # ------------------------------------------------------------
    
    sf = labels.detach().cpu().numpy().squeeze()
    sf = sf*0.5+0.5 
    # Undo normalization
    sf = sf * (max_val - min_val) + min_val
    
    # Convert from log10(a) to a
    sf = 10**sf
    
    unique_a = np.unique(sf)
    
    print(
        f"Available scale-factor range: "
        f"{unique_a.min():.8f} -- {unique_a.max():.8f}"
    )
    
    # ------------------------------------------------------------
    # Select 4 ACTUAL values from unique_a spanning the range
    # ------------------------------------------------------------
    
    target_a = np.linspace(
        unique_a.min(),
        unique_a.max(),
        n_examples
    )
    
    # For each target, choose the closest ACTUAL value in unique_a
    selected_a = np.array([
        unique_a[np.argmin(np.abs(unique_a - target))]
        for target in target_a
    ])
    
    
    print("\nSelected scale factors:")
    for a in selected_a:
        print(f"  {a:.8f}")
    
    
    # ------------------------------------------------------------
    # Find ALL dataset indices belonging to these scale factors
    # ------------------------------------------------------------
    
    candidate_indices = []
    
    for a in selected_a:
    
        matches = np.where(np.isclose(sf, a, rtol=0, atol=1e-12))[0]
    
        print(
            f"a = {a:.8f}: "
            f"{len(matches)} realizations"
        )
    
        candidate_indices.extend(matches)
    
    candidate_indices = np.asarray(candidate_indices)
    
    
    # ------------------------------------------------------------
    # Load all candidate samples
    # ------------------------------------------------------------
    
    sampled_labels = (
        labels[candidate_indices]
        .clone()
        .detach()
        .to(torch.float32)
    )
    
    real_fdm = (
        X_train[candidate_indices]
        .clone()
        .detach()
        .to(torch.float32)
    )
    
    latent_samples = (
        ics[candidate_indices]
        .clone()
        .detach()
        .to(torch.float32)
    )
    
    ics_samples = (
        ics_real[candidate_indices]
        .clone()
        .detach()
        .to(torch.float32)
    )
    
    
    if torch.cuda.is_available():
        latent_samples = latent_samples.cuda()
        sampled_labels = sampled_labels.cuda()
    
    
    # ------------------------------------------------------------
    # Generate candidates in small batches to avoid CUDA OOM
    # ------------------------------------------------------------
    
    batch_size = 2   # try 1, 2, 4 depending on GPU memory
    
    gen_imgs_list = []
    
    G_model.eval()
    
    with torch.no_grad():
    
        for start in range(0, len(latent_samples), batch_size):
    
            end = min(start + batch_size, len(latent_samples))
    
            latent_batch = latent_samples[start:end]
            label_batch = sampled_labels[start:end]
    
            gen_batch = G_model(
                latent_batch,
                label_batch
            )
    
            # Move to CPU immediately to free GPU memory
            gen_imgs_list.append(
                gen_batch.cpu()
            )
    
            del gen_batch
            del latent_batch
            del label_batch
    
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    gen_imgs = torch.cat(gen_imgs_list, dim=0)
    
    del gen_imgs_list
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



    gen_imgs = 0.5 * gen_imgs + 0.5
    real_fdm = 0.5 * real_fdm + 0.5
    latent_vis = 0.5 * latent_samples + 0.5
    sampled_labels = 0.5*sampled_labels + 0.5
    ics_samples = 0.5*ics_samples +0.5


    gen_imgs_np = gen_imgs.permute(0, 2, 3, 4, 1).cpu().numpy()
    real_fdm_np = real_fdm.permute(0, 2, 3, 4, 1).cpu().numpy()
    latent_np = latent_vis.permute(0, 2, 3, 4, 1).cpu().numpy()
    ics_samples_np = ics_samples.permute(0,2,3,4,1).cpu().numpy()
    sampled_labels_np = sampled_labels.cpu().numpy()

    def inverse_global(y, stats):
        mu, sigma = stats["mu"], stats["sigma"]
        return y * sigma + mu

    gen_imgs_np = inverse_global(gen_imgs_np, stats)
    real_fdm_np = inverse_global(real_fdm_np, stats)
    latent_np = inverse_global(latent_np, stats)
    ics_samples_np = inverse_global(ics_samples_np,stats)
    sampled_labels_np = sampled_labels_np * (max_val-min_val) + min_val

    gen_imgs_np = np.sinh(gen_imgs_np)
    real_fdm_np = np.sinh(real_fdm_np)
    latent_np = np.sinh(latent_np)
    ics_samples_np = np.sinh(ics_samples_np)
    sampled_labels_np = 10**sampled_labels_np

    gen_proj = np.log10(np.mean((gen_imgs_np[..., 0]**2 + gen_imgs_np[..., 1]**2) , axis=3))
    real_proj = np.log10(np.mean((real_fdm_np[..., 0]**2 + real_fdm_np[..., 1]**2) , axis=3))
    latent_proj = np.log10(np.mean((latent_np[..., 0]**2 + latent_np[..., 1]**2), axis=3))


    # Magnitude of complex field
    gen_3d = gen_imgs_np[..., 0]**2 + gen_imgs_np[..., 1]**2
    real_3d = real_fdm_np[..., 0]**2 + real_fdm_np[..., 1]**2
    ics_3d = ics_samples_np[...,0]**2 + ics_samples_np[...,1]**2
    latent_3d = latent_np[...,0]**2 + latent_np[...,1]**2
    gen_ps_list = []
    real_ps_list = []
    lin_ps_list = []
    kJ_list = []
    a_0 = 1/(1+127)
    # ------------------------------------------------------------
    # Compute power-spectrum errors
    # ------------------------------------------------------------

    pk_errors = []

    eps = 1e-20

    for i in range(len(gen_3d)):

        gen_delta = density_to_contrast(gen_3d[i])
        real_delta = density_to_contrast(real_3d[i])

        k, ps_gen = power_spectrum_3d(gen_delta, 1.0)
        _, ps_real = power_spectrum_3d(real_delta, 1.0)

        # Optional: focus on nonlinear scales only
        # mask = k > 1.0

        score = np.mean(
            (
                np.log10(ps_gen + eps)
                - np.log10(ps_real + eps)
            )**2
            # [mask]
        )

        pk_errors.append(score)

    pk_errors = np.asarray(pk_errors)


    # ------------------------------------------------------------
    # Select best PS-error realization at each a,
    # while enforcing DIFFERENT realizations
    # ------------------------------------------------------------

    chosen_idx = []
    used_realizations = set()

    n_a = 51

    for a in selected_a:

        # All candidates at this scale factor
        local_idx = np.where(
            np.isclose(
                sf[candidate_indices],
                a,
                rtol=0,
                atol=1e-12
            )
        )[0]

        if len(local_idx) == 0:
            print(
                f"WARNING: no realization found for a={a:.8f}"
            )
            continue

        # Sort candidates from best to worst PS error
        ranked_local_idx = local_idx[
            np.argsort(pk_errors[local_idx])
        ]

        # --------------------------------------------------------
        # Find the best candidate whose realization has not
        # already been used
        # --------------------------------------------------------

        selected_local_idx = None

        for candidate in ranked_local_idx:

            original_idx = candidate_indices[candidate]

            realization_id = original_idx // n_a

            if realization_id not in used_realizations:

                selected_local_idx = candidate
                used_realizations.add(realization_id)

                break

        # --------------------------------------------------------
        # Safety fallback
        # --------------------------------------------------------

        if selected_local_idx is None:

            print(
                f"WARNING: could not find unused realization "
                f"for a={a:.8f}"
            )

            selected_local_idx = ranked_local_idx[0]

        chosen_idx.append(selected_local_idx)

    chosen_idx = np.asarray(chosen_idx)


    
    # ------------------------------------------------------------
    # Print final selection
    # ------------------------------------------------------------
    
    print("\nFinal Figure 5 selection:")
    

    for local_idx in chosen_idx:
    
        original_idx = candidate_indices[local_idx]
    
        # If data are ordered as realization * n_a + a_idx
        realization_id = original_idx // 51
        a_idx = original_idx % 51
    
        print(
            f"dataset index = {original_idx:5d}, "
            f"realization = {realization_id:2d}, "
            f"a_idx = {a_idx:2d}, "
            f"a = {sf[original_idx]:.8f}, "
            f"PS error = {pk_errors[local_idx]:.6e}"
        )

    print(chosen_idx)
    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    n_examples = len(chosen_idx)


    fig, axs = plt.subplots(
        n_examples,
        4,
        figsize=(18, 4*n_examples),
        gridspec_kw={
            "width_ratios": [1, 1, 1, 1.5]
        }
    )
    if n_examples == 1:
        axs = axs[None, :]

    zoom_frac = 0.20#0.20      # zoom window = 20% of image size

    for row, idx in enumerate(chosen_idx):

        inp = latent_proj[idx]
        gen = gen_proj[idx]
        real = real_proj[idx]

        scale_factor = sampled_labels_np[idx,0]#scale_factors[idx]


        H, W = real.shape

        zoom_frac = 0.15

        dh = int(H * zoom_frac / 2)
        dw = int(W * zoom_frac / 2)

        cy = H // 2
        cx = W // 2

        ys = cy - dh
        ye = cy + dh

        xs = cx - dw
        xe = cx + dw


        # --------------------------------------------------------
        # Shared color scaling
        # --------------------------------------------------------
        vmin = min(real.min(), gen.min())
        vmax = max(real.max(), gen.max())

        # --------------------------------------------------------
        # Input
        # --------------------------------------------------------
        ax = axs[row, 0]

        if row == 0:
            ax.text(
                 0.02, 0.96, "LR",
                 transform=ax.transAxes,
                 color="white",
                 fontsize=20,
                 fontweight="bold",
                 va="top"
             )
        ax.imshow(
            inp,
            cmap="inferno",
            origin="lower"
        )


        if row == 0:
            ax.set_title("Input", fontsize=18)
        ax.axis("off")
        # --------------------------------------------------------
        # Generated
        # --------------------------------------------------------
        ax = axs[row, 1]

        ax.imshow(
            gen,
            cmap="inferno",
            origin="lower",
            vmin=vmin,
            vmax=vmax
        )
        if row == 0:
            ax.text(
                 0.02, 0.96, "SR",
                 transform=ax.transAxes,
                 color="white",
                 fontsize=20,
                 fontweight="bold",
                 va="top"
             )
        #ax.set_title(
        #    f"Generated\nSSIM = {scores[idx]:.3f}"
        #)
        if row == 0:
             ax.set_title("Generated", fontsize=18)
        ax.axis("off")

        #axins = ax.inset_axes(
        #    [0.55, 0.55, 0.4, 0.4]
        #)
        axins = ax.inset_axes([0.62, 0.58, 0.33, 0.33])

        axins.imshow(
            gen,
            cmap="inferno",
            origin="lower",
            vmin=vmin,
            vmax=vmax
        )
        for spine in axins.spines.values():
            spine.set_edgecolor("white")
        axins.set_xlim(xs, xe)
        axins.set_ylim(ys, ye)

        axins.set_xticks([])
        axins.set_yticks([])

        ax.indicate_inset_zoom(
            axins,
            edgecolor="white"
        )

        # --------------------------------------------------------
        # Real
        # --------------------------------------------------------
        ax = axs[row, 2]

        ax.imshow(
            real,
            cmap="inferno",
            origin="lower",
            vmin=vmin,
            vmax=vmax
        )

        if row == 0:
            ax.text(
                 0.02, 0.96, "HR",
                 transform=ax.transAxes,
                 color="white",
                 fontsize=20,
                 fontweight="bold",
                 va="top"
             )

        #ax.set_title("Real")
        if row == 0:
           ax.set_title("Real", fontsize=18)
        ax.axis("off")

        #axins = ax.inset_axes(
        #    [0.55, 0.55, 0.4, 0.4]
        #)
        axins = ax.inset_axes([0.62, 0.58, 0.33, 0.33])

        axins.imshow(
            real,
            cmap="inferno",
            origin="lower",
            vmin=vmin,
            vmax=vmax
        )
        for spine in axins.spines.values():
            spine.set_edgecolor("white")
        axins.set_xlim(xs, xe)
        axins.set_ylim(ys, ye)

        axins.set_xticks([])
        axins.set_yticks([])

        ax.indicate_inset_zoom(
            axins,
            edgecolor="white"
        )
        ## Power spectrum
        gen_delta  = density_to_contrast(gen_3d[idx])
        real_delta = density_to_contrast(real_3d[idx])
        ic_delta   = density_to_contrast(ics_3d[idx])
        lat_delta  = density_to_contrast(latent_3d[idx])
        k, ps_gen = power_spectrum_3d(gen_delta, 1.0)
        _, ps_real = power_spectrum_3d(real_delta, 1.0)
        _, ps_ic = power_spectrum_3d(ic_delta, 1.0)
        k_lat, ps_lat = power_spectrum_3d(lat_delta, 1.0)

        a = sampled_labels_np[idx][0]

        T, kJ = T_fdm_approx(
            k,
            a,
            m22,
            Omega_m_h2
        )

        ps_lin = (
            a**2 / a_0**2
            * ps_ic
            * T**2
        )
        print('a:',a)

        ratio = ps_gen / (ps_real + 1e-30)
        ax = axs[row, 3]

        # parent subplot location
        parent_spec = axs[row, 3].get_subplotspec()

        # split into power spectrum + ratio
        subspec = GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=parent_spec,
            height_ratios=[3, 1],
            hspace=0.05
        )

        # remove original axes occupying that slot
        axs[row, 3].remove()

        ax_ps = fig.add_subplot(subspec[0])
        ax_ratio = fig.add_subplot(subspec[1], sharex=ax_ps)

        # ---------- Power spectrum ----------
        ax_ps.loglog(k, k**3 * ps_gen, lw=2, label="SR")
        ax_ps.loglog(k, k**3 * ps_real, "--", lw=2, label="HR")
        ax_ps.loglog(k_lat, k_lat**3 * ps_lat, "--", lw=2, label="LR")
        ax_ps.loglog(k, k**3 * ps_lin, ":", lw=2, label="Linear")

        ax_ps.axvline(kJ, color="r", alpha=0.7, label=r"$k_J$")

        ax_ps.set_ylabel(r"$k^3P(k)$", fontsize=14)

        if row == 0:
            ax_ps.legend(fontsize=14, frameon=False)
            ax_ps.set_title("Power Spectrum",fontsize = 16)

        # ---------- Ratio ----------
        #ratio_sr = ps_real / (ps_gen)
        #ratio_lr = ps_lat / (ps_real + 1e-30)
        frac_err = ps_gen / ps_real - 1
        ax_ratio.semilogx(k, frac_err, lw=2)
        #ax_ratio.semilogx(k_lat, ratio_lr, lw=2)

        ax_ratio.axhline(0.0, color="k", ls="--", alpha=0.5)

        ax_ratio.axvline(kJ, color="r", alpha=0.7)

        ax_ratio.set_ylim(-0.5, 0.5)     # adjust as needed
        ax_ratio.set_ylabel("Error (SR/HR-1)")
        if row == 3:
            ax_ratio.set_xlabel(r"$k\,[h\,{\rm Mpc}^{-1}]$", fontsize=14)

        plt.setp(ax_ps.get_xticklabels(), visible=False)

    # ------------------------------------------------------------
    # Save figure
    # ------------------------------------------------------------
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            f"Fig6/7.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()







def sample_images2(epoch, G_model, X_train, labels, ics,ics_real, stats,max_val, min_val, save_dir="./superres_reals_res_pi"):
    os.makedirs(save_dir, exist_ok=True)

    r, c = 4, 6  # 2 samples per row, each sample uses 3 columns
    samples_per_row = 4#c // 3
    a_idx = 25      # middle redshift
    #np.random.seed(21)
    #index = #np.random.randint(0, len(X_train), r * samples_per_row)
    n_realizations = 16
    n_a = 51
    
    index = np.array([
        r*n_a + a_idx
        for r in range(n_realizations)
    ])
    sampled_labels = labels[index].clone().detach().to(torch.float32)
    real_fdm = X_train[index].clone().detach().to(torch.float32)
    latent_samples = ics[index].clone().detach().to(torch.float32)
    ics_samples = ics_real[index].clone().detach().to(torch.float32)
    if torch.cuda.is_available():
        latent_samples = latent_samples.cuda()
        sampled_labels = sampled_labels.cuda()

    G_model.eval()
    with torch.no_grad():
        gen_imgs = G_model(latent_samples, sampled_labels)
    G_model.train()

    gen_imgs = 0.5 * gen_imgs + 0.5
    real_fdm = 0.5 * real_fdm + 0.5
    latent_vis = 0.5 * latent_samples + 0.5
    sampled_labels = 0.5*sampled_labels + 0.5
    ics_samples = 0.5*ics_samples +0.5


    gen_imgs_np = gen_imgs.permute(0, 2, 3, 4, 1).cpu().numpy()
    real_fdm_np = real_fdm.permute(0, 2, 3, 4, 1).cpu().numpy()
    latent_np = latent_vis.permute(0, 2, 3, 4, 1).cpu().numpy()
    ics_samples_np = ics_samples.permute(0,2,3,4,1).cpu().numpy()
    sampled_labels_np = sampled_labels.cpu().numpy()

    def inverse_global(y, stats):
        mu, sigma = stats["mu"], stats["sigma"]
        return y * sigma + mu

    gen_imgs_np = inverse_global(gen_imgs_np, stats)
    real_fdm_np = inverse_global(real_fdm_np, stats)
    latent_np = inverse_global(latent_np, stats)
    ics_samples_np = inverse_global(ics_samples_np,stats)
    sampled_labels_np = sampled_labels_np * (max_val-min_val) + min_val

    gen_imgs_np = np.sinh(gen_imgs_np)
    real_fdm_np = np.sinh(real_fdm_np)
    latent_np = np.sinh(latent_np)
    ics_samples_np = np.sinh(ics_samples_np)
    sampled_labels_np = 10**sampled_labels_np

    gen_proj = np.log10(np.mean((gen_imgs_np[..., 0]**2 + gen_imgs_np[..., 1]**2) , axis=3))
    real_proj = np.log10(np.mean((real_fdm_np[..., 0]**2 + real_fdm_np[..., 1]**2) , axis=3))
    latent_proj = np.log10(np.mean((latent_np[..., 0]**2 + latent_np[..., 1]**2), axis=3))


    # Magnitude of complex field
    gen_3d = gen_imgs_np[..., 0]**2 + gen_imgs_np[..., 1]**2
    real_3d = real_fdm_np[..., 0]**2 + real_fdm_np[..., 1]**2
    ics_3d = ics_samples_np[...,0]**2 + ics_samples_np[...,1]**2
    latent_3d = latent_np[...,0]**2 + latent_np[...,1]**2
    gen_ps_list = []
    real_ps_list = []
    lin_ps_list = []
    kJ_list = []
    a_0 = 1/(1+127)


    # ==========================================================
    # Compute power spectra for ALL selected realizations
    # ==========================================================
    
    ratio_list = []
    frac_error_list = []
    
    for idx in range(len(gen_3d)):
        print(sampled_labels_np[idx,0])  
        gen_delta  = density_to_contrast(gen_3d[idx])
        real_delta = density_to_contrast(real_3d[idx])
    
        k, ps_gen = power_spectrum_3d(gen_delta, 1.0)
        _, ps_real = power_spectrum_3d(real_delta, 1.0)
    
        ratio = ps_gen / (ps_real)
    
        ratio_list.append(ratio)
        frac_error_list.append(ratio - 1.0)
    
    ratio_arr = np.asarray(ratio_list)          # (Nrealizations, Nk)
    frac_arr  = np.asarray(frac_error_list)
    #keep = np.arange(len(ratio_arr))
    frac_arr  = frac_arr[keep] 
    mean_ratio = np.mean(ratio_arr, axis=0)
    std_ratio  = np.std(ratio_arr, axis=0)
    
    mean_frac = np.mean(frac_arr, axis=0)
    std_frac  = np.std(frac_arr, axis=0)
    
    # ==========================================================
    # Per-realization quality score
    # ==========================================================
    
    scores = np.mean(np.abs(frac_arr), axis=1)
    
    print("\n====================================")
    print("Power Spectrum Error Summary")
    print("====================================")
    print("Mean realization error:", scores.mean())
    print("Std realization error :", scores.std())
    
    order = np.argsort(scores)
    
    print("\nBest realizations:")
    for i in order[:5]:
        print(f"{i:3d}  error={scores[i]:.4f}")
    
    print("\nWorst realizations:")
    for i in order[-5:]:
        print(f"{i:3d}  error={scores[i]:.4f}")
    
    lat_delta  = density_to_contrast(latent_3d[0])
    k_lat, ps_lat = power_spectrum_3d(lat_delta, 1.0)

    a = sampled_labels_np[0][0]

    T, kJ = T_fdm_approx(
        k,
        a,
        m22,
        Omega_m_h2
    )


    
    # ==========================================================
    # Fractional error (%) + 1 sigma
    # ==========================================================
    
    plt.figure(figsize=(7,5))

    plt.semilogx(
        k,
         mean_frac,
        lw=3,
        label='Mean'
    )
    
    plt.fill_between(
        k,
         (mean_frac - std_frac),
         (mean_frac + std_frac),
        alpha=0.3,
        label=r'$\pm1\sigma$'
    )
    
    plt.axhline(
        0.0,
        color='k',
        ls='--'
    )
   
    plt.axvline(k_lat[-1], color='k', ls='--', lw=1.5,
               label=r'$k_{\rm LR}$')
    
    plt.axvline(kJ, color='firebrick', ls=':',
               lw=1.5,
               label=r'$k_{\rm J}$')
    plt.ylim(-1.0, 2.0) 
    plt.xlabel(r'$k\,[h\,{\rm Mpc}^{-1}]$',fontsize=18)
    plt.ylabel('Fractional Power Spectrum Error ',fontsize =14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=10)
    plt.title('SR model (with SP loss)', fontsize=14, fontweight='bold')
    #plt.title('SR model (with KDK loss)', fontsize=14, fontweight='bold')
    plt.legend(fontsize = 10)
    plt.tight_layout()
    
    plt.savefig(
        os.path.join(save_dir, f"Fig8a/8b.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()



##################
# Main
##################
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")


    save_dir1 = "link to the path for SR_DATA and fdm_test3.npy from Zenodo"
    

    fdm_test = np.load(os.path.join(save_dir1, "fdm_test3.npy"))
    labels_test = np.load(os.path.join(save_dir1, "labels_test3.npy"))
    ic_test_lr = np.load(os.path.join(save_dir1, "ic_test3_lr.npy"))
    ic_test_real = np.load(os.path.join(save_dir1, "ic_test3.npy"))


    stats = np.load(os.path.join(save_dir1, "stats.npy"),allow_pickle = True).item()
    # ---- convert to torch ----
    max_val = np.load(os.path.join(save_dir1,"max_val.npy"))
    min_val = np.load(os.path.join(save_dir1,"min_val.npy"))



    fdm_test = torch.tensor(fdm_test, dtype=torch.float32)
    labels_test = torch.tensor(labels_test, dtype=torch.float32)
    ic_test_real = torch.tensor(ic_test_real, dtype = torch.float32)
    ic_test = torch.tensor(ic_test_lr, dtype = torch.float32)

    print("Initializing generator...")
    G = ConditionalGenerator(cond_dim=64).to(device)
    # Either load SR_model trained with SP loss
    #G.load_state_dict(torch.load('https://github.com/akmdevx/DeepWaves.Cosmo-SPINN/releases/PI_SR_SP_final.pt'))
    # Or load SR_model trained with KDK loss
    G.load_state_dict(torch.load('https://github.com/akmdevx/DeepWaves.Cosmo-SPINN/releases/PI_SR_FO_final.pt'))

    # Either this function to generate Fig 6 or 7 of the paper
    sample_images3(510,G,fdm_test2,labels_test2, ic_test2,ic_test2_real, stats,max_val,min_val)    
    # or this function to generate Fig 8a or 8b of the paper
    sample_images2(510,G,fdm_test2,labels_test2, ic_test2,ic_test2_real, stats,max_val,min_val)    
