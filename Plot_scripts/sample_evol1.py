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

from ./Models import Evol_net.py


from numpy.random import randint



##################
# Visualization
##################



def sample_fields_re(
    epoch,
    G_model,
    X_train,
    labels,
    ics,
    stats,
    idx=None,
    save_dir="./evol_pi/"
):

    os.makedirs(save_dir, exist_ok=True)


    a_min_l = -0.8273942837611394
    a_max_l = -0.8048255392272029

    # build scale factors 
    labels_norm = (labels * 0.5 + 0.5)
    labels_norm = labels_norm * (a_max_l - a_min_l)+a_min_l
    a_act = 10 ** labels_norm  # shape (B,)
    tol = 0.005
    target_a =0.1540 

    idx = torch.argmin(torch.abs(a_act - target_a)).item()
    sampled_labels = labels[idx:idx+1].clone().detach().to(torch.float32)
    real_fdm = X_train[idx:idx+1].clone().detach().to(torch.float32)
    latent_samples = ics[idx:idx+1].clone().detach().to(torch.float32)

    if torch.cuda.is_available():
        latent_samples = latent_samples.cuda()
        sampled_labels = sampled_labels.cuda()

    G_model.eval()

    with torch.no_grad():
        gen_imgs = G_model(latent_samples, sampled_labels)

    G_model.train()

    # =====================================================
    # EXACTLY SAME PIPELINE AS ORIGINAL FUNCTION
    # =====================================================

    gen_imgs = 0.5 * gen_imgs + 0.5
    real_fdm = 0.5 * real_fdm + 0.5
    latent_samples = 0.5*latent_samples + 0.5

    gen_imgs_np = (
        gen_imgs.permute(0, 2, 3, 4, 1)
        .cpu()
        .numpy()
    )

    real_fdm_np = (
        real_fdm.permute(0, 2, 3, 4, 1)
        .cpu()
        .numpy()
    )
    latent_np = (
        latent_samples.permute(0, 2, 3, 4, 1)
        .cpu()
        .numpy()
    )

    def inverse_global(y, stats):
        mu, sigma = stats["mu"], stats["sigma"]
        return y * sigma + mu

    gen_imgs_np = inverse_global(gen_imgs_np, stats)
    real_fdm_np = inverse_global(real_fdm_np, stats)
    latent_np = inverse_global(latent_np, stats)

    gen_imgs_np = np.sinh(gen_imgs_np)
    real_fdm_np = np.sinh(real_fdm_np)
    latent_np = np.sinh(latent_np)
    # =====================================================

    pred = gen_imgs_np[0]
    real = real_fdm_np[0]
    lat = latent_np[0]
    # -----------------------------------------------------
    # Projected quantities
    # -----------------------------------------------------

    R_pred = np.mean(np.abs(pred[..., 0]), axis=2)
    R_real = np.mean(np.abs(real[..., 0]), axis=2)
    R_lat = np.mean(np.abs(lat[..., 0]), axis = 2)

    I_pred = np.mean(np.abs(pred[..., 1]), axis=2)
    I_real = np.mean(np.abs(real[..., 1]), axis=2)
    I_lat  = np.mean(np.abs(lat[..., 1]), axis =2)

    rho_pred = pred[..., 0]**2 + pred[..., 1]**2
    rho_real = real[..., 0]**2 + real[..., 1]**2
    rho_lat = lat[..., 0]**2 + lat[...,1]**2

    rho_pred = np.log10(
        np.mean(rho_pred, axis=2) + 1e-12
    )

    rho_real = np.log10(
        np.mean(rho_real, axis=2) + 1e-12
    )

    rho_init = np.log10(
        np.mean(rho_lat, axis=2) + 1e-12
    )


    fields = [
        (R_real, R_pred, r"$\langle |R| \rangle_z$"),
        (I_real, I_pred, r"$\langle |I| \rangle_z$"),
        (rho_real, rho_pred, r"$\log_{10}\langle \rho \rangle_z$")
    ]

    latent_fields = [
        R_lat,
        I_lat,
        rho_init,
    ]
    
    latent_labels = [
        r"$\langle |R| \rangle_z$",
        r"$\langle |I| \rangle_z$",
        r"$\log_{10}\langle \rho \rangle_z$",
    ]
    
    fig, axs = plt.subplots(
        3,
        4,
        figsize=(16, 12)
    )
    
    title_fs = 24
    label_fs = 24
    cbar_fs = 18
    
    for row, ((truth, pred, label), latent, latent_label) in enumerate(
        zip(fields, latent_fields, latent_labels)
    ):
    
        residual = pred - truth
    
        vmin = min(truth.min(), pred.min())
        vmax = max(truth.max(), pred.max())
    
        vmax_res = np.max(np.abs(residual))
    
        im_lat = axs[row, 0].imshow(
            latent,
            cmap="inferno",
            vmin = vmin,
            vmax = vmax
        )
    
        im0 = axs[row, 1].imshow(
            truth,
            cmap="inferno",
            vmin=vmin,
            vmax=vmax
        )
    
        im1 = axs[row, 2].imshow(
            pred,
            cmap="inferno",
            vmin=vmin,
            vmax=vmax
        )
    
        im2 = axs[row, 3].imshow(
            residual,
            cmap="RdBu_r",
            vmin=-vmax_res,
            vmax=vmax_res
        )
    
        axs[row, 0].set_ylabel(
            latent_label,
            fontsize=label_fs
        )
    
        if row == 0:
            axs[row, 0].set_title("Initial condition", fontsize=title_fs)
            axs[row, 1].set_title("Truth", fontsize=title_fs)
            axs[row, 2].set_title("Prediction", fontsize=title_fs)
            axs[row, 3].set_title("Residual", fontsize=title_fs)
    
        for col in range(4):
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
    
        cbar_lat = plt.colorbar(
            im_lat,
            ax=axs[row, 0],
            fraction=0.046,
            pad=0.04
        )
        cbar_lat.ax.tick_params(labelsize=cbar_fs)
    
        cbar0 = plt.colorbar(
            im0,
            ax=axs[row, 1],
            fraction=0.046,
            pad=0.04
        )
        cbar0.ax.tick_params(labelsize=cbar_fs)
    
        cbar1 = plt.colorbar(
            im1,
            ax=axs[row, 2],
            fraction=0.046,
            pad=0.04
        )
        cbar1.ax.tick_params(labelsize=cbar_fs)
    
        cbar2 = plt.colorbar(
            im2,
            ax=axs[row, 3],
            fraction=0.046,
            pad=0.04
        )
        cbar2.ax.tick_params(labelsize=cbar_fs)
    
    plt.tight_layout()

    a_val = sampled_labels[0,0].detach().cpu().numpy()
    a_min_l = -0.8273942837611394
    a_max_l = -0.8048255392272029

    # build scale factors a, a2 (keep them 1D arrays for compute_dt)
    labels_proc = (a_val * 0.5 + 0.5)
    labels_proc = labels_proc * (a_max_l - a_min_l)+a_min_l
    a = 10 ** labels_proc  # shape (B,)
    print(a)
    plt.suptitle(
        rf"$a = {a:.3f}$",
        fontsize=24
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            f"Fig4.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )






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

def density_to_contrast(rho):
    mean = np.mean(rho)
    return (rho - mean) / mean

def evaluate_power_spectrum_evolution(epoch,
    G_model,
    X_test,
    labels,
    ics,
    stats,
    save_dir="./pi_10/"
):

    os.makedirs(save_dir, exist_ok=True)

    a_values = []
    ps_error = []
    ps_error_std = []
    G_model.eval()

    with torch.no_grad():

        for i in range(len(X_test)):

            label = labels[i:i+1].clone().detach().float()
            latent = ics[i:i+1].clone().detach().float()
            real = X_test[i:i+1].clone().detach().float()

            if torch.cuda.is_available():
                label = label.cuda()
                latent = latent.cuda()

            pred = G_model(latent, label)

            # ======================================
            # EXACT SAME PIPELINE AS SAMPLER
            # ======================================

            pred = 0.5 * pred + 0.5
            real = 0.5 * real + 0.5

            pred = pred.permute(
                0,2,3,4,1
            ).cpu().numpy()

            real = real.permute(
                0,2,3,4,1
            ).cpu().numpy()

            def inverse_global(y, stats):
                mu, sigma = stats["mu"], stats["sigma"]
                return y * sigma + mu

            pred = inverse_global(pred, stats)
            real = inverse_global(real, stats)

            pred = np.sinh(pred)
            real = np.sinh(real)

            pred = pred[0]
            real = real[0]

            # ======================================

            rho_pred = (
                pred[...,0]**2 +
                pred[...,1]**2
            )

            rho_real = (
                real[...,0]**2 +
                real[...,1]**2
            )
            pred_del = density_to_contrast(rho_pred)
            real_del = density_to_contrast(rho_real)
            k_pred, P_pred = power_spectrum_3d(pred_del,1.0)
            k_real, P_real = power_spectrum_3d(real_del,1.0)

            mask = P_real > 0

            ratio = (
                P_pred[mask] /
                P_real[mask]
            )

            k_error = np.abs(ratio - 1.0)
            
            err_mean = np.mean(k_error)
            err_std = np.std(k_error)
            
            ps_error.append(err_mean)
            ps_error_std.append(err_std)



            a_min_l = -0.8273942837611394
            a_max_l = -0.8048255392272029

            # build scale factors a, a2 (keep them 1D arrays for compute_dt)
            labels_proc = (label * 0.5 + 0.5)
            labels_proc = labels_proc * (a_max_l - a_min_l)+a_min_l
            a = 10 ** labels_proc  # shape (B,)

            a_values.append(
                float(
                    a.detach()
                    .cpu()
                    .numpy()
                    .squeeze()
                )
            )

    #a_values = np.array(a_values)
    #ps_error = np.array(ps_error)

    a_values = np.array(a_values)
    ps_error = np.array(ps_error)
    ps_error_std = np.array(ps_error_std)
    
    order = np.argsort(a_values)
    
    a_values = a_values[order]
    ps_error = ps_error[order]
    ps_error_std = ps_error_std[order]
    
    ps_error_upper = ps_error + ps_error_std
    ps_error_lower = np.maximum(
        ps_error - ps_error_std,
        0
    )
    
    plt.figure(figsize=(7,5))
    
    plt.plot(
        a_values,
        ps_error,
        lw=2,
        label="Mean"
    )
    
    plt.fill_between(
        a_values,
        ps_error_lower,
        ps_error_upper,
        alpha=0.25,
        label=r"$\pm1\sigma$ over $k$"
    )
    
    plt.axvspan(
        a_values.min(),
        a_values[15],
        alpha=0.2,
        color="gray"
    )
    
    plt.axvspan(
        a_values[-15],
        a_values.max(),
        alpha=0.2,
        color="gray"
    )
    
    plt.ylim(0, 0.8)
    
    plt.xlabel("Scale Factor $a$", fontsize=14)
    plt.ylabel(
        r"$\langle |P_{\rm pred}/P_{\rm true}-1| \rangle_k$", fontsize=14
    )
    plt.tick_params(axis="both", labelsize=12)
    plt.legend()
    
    plt.tight_layout()
    
    plt.savefig(
        os.path.join(
            save_dir,
            "Fig5.png"
        ),
    dpi=300
    )
    
    plt.close()



##################
# Main
##################
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    save_dir1 = "link to the path for EVOL_DATA_SR from Zenodo"
    fdm_train_np = np.load(os.path.join(save_dir1, "fdm_train.npy"))
    labels_train_np = np.load(os.path.join(save_dir1, "labels_train.npy"))
    ic_train_np = np.load(os.path.join(save_dir1, "ic_train.npy"))

    fdm_test_np = np.load(os.path.join(save_dir1, "fdm_test.npy"))
    labels_test_np = np.load(os.path.join(save_dir1, "labels_test.npy"))
    ic_test_np = np.load(os.path.join(save_dir1, "ic_test.npy"))

    stats = np.load(os.path.join(save_dir1, "stats.npy"),allow_pickle = True).item()
    # ---- convert to torch ----
    fdm_train = torch.tensor(fdm_train_np, dtype=torch.float32)
    labels_train = torch.tensor(labels_train_np, dtype=torch.float32)
    ic_train = torch.tensor(ic_train_np, dtype=torch.float32)
    #ic_train_real = torch.tensor(ic_train_np_real, dtype = torch.float32)
    max_val = np.load(os.path.join(save_dir1,"max_val.npy"))
    min_val = np.load(os.path.join(save_dir1,"min_val.npy"))
    #print(fdm_train.shape)
    #print(labels_train.shape)
    #print(ic_train.shape)
    #print(ic_train_real.shape)

    print(max_val)
    print(min_val)
    fdm_test = torch.tensor(fdm_test_np, dtype=torch.float32)
    labels_test = torch.tensor(labels_test_np, dtype=torch.float32)
    ic_test = torch.tensor(ic_test_np, dtype=torch.float32)


    labels = torch.cat([labels_train, labels_test], dim=0)
    ic_data = torch.cat([ic_train, ic_test], dim=0)


    
    # Initialize model
    print("Initializing generator...")
    G = ConditionalGenerator(cond_dim=64).to(device)

      
    G.load_state_dict(torch.load('https://github.com/akmdevx/DeepWaves.Cosmo-SPINN/releases/PI_Evol_Mod.pt')) 


    # Use sample_fields_re function to get the Fig 4 of the paper
    sample_fields_re(2010,G,fdm_test,labels_test, ic_test, stats)

    fdm_data = torch.cat([fdm_train, fdm_test], dim=0)
    labels = torch.cat([labels_train, labels_test], dim=0)
    ic_data = torch.cat([ic_train, ic_test], dim=0)
    #Use evaluate_power_spectrum_evolution to get the Fig 5 of the paper
    evaluate_power_spectrum_evolution(2010,G,fdm_data,labels, ic_data, stats)
    
 
