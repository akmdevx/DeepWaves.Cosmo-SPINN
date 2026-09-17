# Plot Scripts

This folder contains the scripts and data links required to reproduce the figures presented in the paper:

**[Cosmo-SPINN: Physics-Informed Generative Modeling of Fuzzy Dark Matter Simulations](https://arxiv.org/abs/2607.28604)**

The following sections provide a figure-by-figure guide to reproducing the plots.

## Figure Reproducibility

### Figure 1

Figure 1 was created manually by the author using [draw.io](https://www.drawio.com).

### Figures 2 and 3: Loss Plots

Figures 2 and 3 show the training and evaluation loss curves. These figures can be reproduced using the following scripts and the data provided in the `Loss_data/` folder.

| Figure  | Script                   |
| ------- | ------------------------ |
| Fig. 2a | `plot_eval_data_loss.py` |
| Fig. 2b | `plot_eval_SP_loss.py`   |
| Fig. 3a | `plot_SR_SP_loss.py`     |
| Fig. 3b | `plot_SR_KDK_loss.py`    |

### Figures 4–8: Simulation Results

To reproduce the remaining figures, first download the simulation data from the following Zenodo record:

[Zenodo dataset](https://zenodo.org/uploads/22734983)

The trained model weights are available in the [DeepWaves.Cosmo-SPINN GitHub Releases](https://github.com/akmdevx/DeepWaves.Cosmo-SPINN/releases/).

Download the required model weights and place them in a suitable directory. Update the corresponding paths in the scripts before running them.

The figures can then be reproduced as follows:

| Figures         | Script              |
| --------------- | ------------------- |
| Figures 4 and 5 | `sample_evol1.py`   |
| Figures 6 and 7 | `sample_SR.py`      |
| Figure 8        | `sample_evol_mr.py` |

## Reproducibility Workflow

1. Clone this repository.
2. Install the required dependencies.
3. Use the loss data in `Loss_data/` to reproduce Figures 2 and 3.
4. Download the simulation data from Zenodo.
5. Download the trained model weights from the GitHub Releases page.
6. Update the data and model-weight paths in the sampling scripts.
7. Run the corresponding scripts to reproduce Figures 4–8.

Following this workflow, all figures presented in the paper can be reproduced.

For further details about the methodology and results, please refer to the [Cosmo-SPINN paper](https://arxiv.org/abs/2607.28604).
