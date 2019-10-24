# coding: utf-8

# ===================================================================
# Generates plots illustrating a decoding model performance
# Romuald Menuet - May 2019
# ===================================================================
# Summary: Scripts that generates a performance plot
#          as in TODO:ajouter_nom_article
# ===================================================================

import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
import matplotlib.lines as mlines
from meta_fmri.explore.plotting import plot_embedded
import torch
import nibabel as nib
from nilearn.input_data import NiftiMasker
from .b_decoding_experiment import PytorchEstimator


def wrap(text, length):
    """ Helper function to wrap text with '—' """
    return (
        "—" * ((length - len(text)) // 2)
        + " " + text + " "
        + "—" * ((length - len(text)) // 2)
        + "—" * ((length - len(text)) % 2)
    )


def main():
    # --------------
    # --- CONFIG ---
    # --------------
    parser = argparse.ArgumentParser(
        description="A script to plot results from a decoding model",
        epilog='''Examples:
        - python perf_plot.py -p per_label_results.csv -r perf.PNG'''
    )
    # TODO: combine and replace by just the experiment config
    parser.add_argument("-p", "--perf_file",
                        default="../Data/results/per_label_results.csv",
                        help="path to the CSV file with the per-label "
                             "performance")
    parser.add_argument("-r", "--results_file",
                        default="../Data/results/perf.PNG",
                        help="Path to the file where results are saved")
    parser.add_argument("-f", "--features",
                        default="../Data/X.p",
                        help="Path of the pickle file "
                             "containing the matrix of fMRI stat maps")
    parser.add_argument("-m", "--mask",
                        default="../Data/masks/mask.nii.gz",
                        help="Path of the Nifti file containing the mask used "
                             "for full voxel maps")
    parser.add_argument("-a", "--atlas",
                        default="../Data/models/components_1024_task.nii.gz",
                        help="Path of the Nifti file containing the brain atlas"
                             " (dictionary) used to embed the features used "
                             "for both encoding and decoding")
    parser.add_argument("-l", "--labels",
                        default="../Data/Y.p",
                        help="Path of the pickle file "
                             "containing the matrix of map labels")
    parser.add_argument("-d", "--decoding_model",
                        default="../Data/models/clf.pt",
                        help="Path of the pytorch dump of a decoding model "
                             "trained provided on the maps and labels")
    parser.add_argument("-e", "--encodings",
                        default="../Data/models/encodings.p",
                        help="Path of the pickle dump of the dictionary "
                             "of encoding maps")
    parser.add_argument("-c", "--concepts",
                        default="../Data/concepts.csv",
                        help="Path of the CSV file of comma separated concepts"
                             ", ordered as the columns of the labels file")
    parser.add_argument("-r", "--results_file",
                        default="../Data/results/maps.PNG",
                        help="Path to the file where results are saved")

    args = parser.parse_args()

    # --------------------
    # --- DATA LOADING ---
    # --------------------
    results_model = pd.read_csv(args.perf_file, index_col=0)
    results_model.sort_values("AUC TEST", ascending=False, inplace=True)
    results_model.index.name = "Concept"

    # ------------------------
    # --- PLOT PERFORMANCE ---
    # ------------------------
    sns.set(style="whitegrid")
    sns.set_context("notebook", font_scale=1.1)
    font = {'fontname': 'DejaVu Sans Mono'}
    ylabels = list(results_model.index)

    ylabels_padded = [" " * 39 + wrap(lab, 37)
                      for lab in ylabels]

    fig, axes = plt.subplots(nrows=1, ncols=3,
                             gridspec_kw={'width_ratios': [10, 18, 40]})
    fig.set_figheight(16)
    fig.set_figwidth(15)
    sns.set_style("whitegrid")

    ind = np.arange(len(results_model))[::-1]
    height_nnod = 0.4
    height_other = 0.15

    # Plot ratio in TRAIN
    i = 0
    sns.set_color_codes("muted")
    axes[i].set(xlim=(0, 0.75),
                ylim=(-1, len(results_model)),
                ylabel="",
                xlabel="ratio in TRAIN dataset")
    axes[i].xaxis.set_major_locator(ticker.MultipleLocator(0.25))
    axes[i].invert_xaxis()
    axes[i].set_yticklabels(np.arange(len(results_model)))
    axes[i].set_yticks([])  # Hide the left y-axis ticks
    axes[i].grid(False)
    ax_twin = axes[i].twinx()  # Create a twin x-axis
    ax_twin.axvline(0.25, 0, 1, color='grey', linestyle='--', linewidth=0.5)
    ax_twin.axvline(0.5, 0, 1, color='grey', linestyle='--', linewidth=0.5)
    ax_twin.barh(ind,
                 results_model["ratio TRAIN"].values,
                 height_nnod + 2 * height_other,
                 linewidth=0,
                 color='darkorange')
    ax_twin.set_yticks(ind)
    ax_twin.set_yticklabels(ylabels_padded,
                            horizontalalignment='center',
                            **font)
    ax_twin.set(ylabel="", ylim=(-1, len(results_model)))
    ax_twin.tick_params(axis=u'both', which=u'both', length=0)
    ax_twin.grid(False)

    # Plot empty
    i = 1
    axes[i].axis("off")
    axes[i].set(ylim=(-1, len(results_model)))

    # Plot AUC
    i = 2
    axes[i].axvline(0.25, 0, 1, color='grey', linestyle='--', linewidth=0.5)
    axes[i].axvline(0.5, 0, 1, color='grey', linestyle='--')
    axes[i].axvline(0.75, 0, 1, color='grey', linestyle='--', linewidth=0.5)

    axes[i].barh(ind,
                 results_model["AUC TEST"].values,
                 height_nnod,
                 linewidth=0,
                 color='darkorange')

    axes[i].set(xlim=(0, 1),
                ylim=(-1, len(results_model)),
                ylabel="",
                xlabel="AUC on IBC dataset")
    axes[i].set_yticks(ind)
    axes[i].set_yticklabels([" "] * len(results_model))
    axes[i].xaxis.set_major_locator(ticker.MultipleLocator(0.25))
    axes[i].grid(False)

    plt.savefig(args.result_file,
                bbox_inches='tight',
                pad_inches=0.1)

    # --------------------
    # --- DATA LOADING ---
    # --------------------
    vocab = list(pd.read_csv(args.concepts, index_col=0).values.flat)

    mask = nib.load(args.maskl)
    masker = NiftiMasker(mask_img=mask).fit()

    atlas = nib.load(args.atlas)
    atlas_masked = masker.transform(atlas)

    with open(args.encodings, 'rb') as f:
        encodings_dict = pickle.load(f)

    with open(args.features, 'rb') as f:
        X = pickle.load(f)

    decoder = PytorchEstimator.from_file(args.decoding_model)

    X_t = torch.tensor(X).float()
    X_t.requires_grad = True
    decoder.model.eval()

    # -------------------------
    # --- PLOT CONSTRUCTION ---
    # -------------------------
    n = len(vocab)
    n_col = 6
    scale_box = 0.98

    box_axes = [None] * n
    title_axes = [None] * n
    enc_axes = [None] * n
    dec_axes = [None] * n
    n_row = (n - 1) // n_col + 1
    width = 1 / n_col
    height = 1 / n_row

    fig = plt.figure(figsize=(4 * n_col, 4 * n_row))
    for i in range(n):
        concept = vocab[i]
        row = i // n_col
        col = i % n_col
        min_x = col * width
        min_y = (n_row - row - scale_box) * height

        box_axes[i] = fig.add_axes([
            min_x,
            min_y,
            width * scale_box,
            height * scale_box
        ])
        box_axes[i].set_xticks([])
        box_axes[i].set_yticks([])
        sep = mlines.Line2D(
            [0.1, 0.9], [0.43, 0.43],
            color='grey', linewidth=1.0
        )
        box_axes[i].add_line(sep)

        title_axes[i] = fig.add_axes([
            min_x,
            min_y + 0.87 * height * scale_box,
            width * scale_box,
            0.13 * height * scale_box
        ])
        title_axes[i].set_xticks([])
        title_axes[i].set_yticks([])
        if len(concept) <= 27:
            title = concept
        else:
            title = concept[:24] + "..."
        title_axes[i].text(
            0.5, 0.5,
            title,
            fontsize=18,
            weight='bold',
            va='center',
            ha='center'
        )

        enc_axes[i] = fig.add_axes([
            min_x + 0.03 * width * scale_box,
            min_y + 0.45 * height * scale_box,
            0.93 * width * scale_box,
            0.40 * height * scale_box
        ])
        enc_axes[i].set_xticks([])
        enc_axes[i].set_yticks([])
        pl_enc = plot_embedded(
            encodings_dict[concept].flatten(),
            atlas_masked, masker,
            plot_type="glass_brain", axes=enc_axes[i]
        )
        pl_enc.title("Enc.", color='k', bgcolor='w', alpha=0, size=16,
                     weight='bold')

        dec_axes[i] = fig.add_axes([
            min_x + 0.03 * width * scale_box,
            min_y,
            0.93 * width * scale_box,
            0.40 * height * scale_box
        ])
        dec_axes[i].set_xticks([])
        dec_axes[i].set_yticks([])
        der = torch.mean(
            torch.autograd.grad(torch.mean(decoder.model(X_t)[:, i]),
                                X_t)[0], 0).detach().numpy()
        pl_dec = plot_embedded(
            der, atlas_masked, masker,
            plot_type="glass_brain", axes=dec_axes[i]
        )
        pl_dec.title("Dec.", color='k', bgcolor='w', size=16, weight='bold')

    plt.savefig(args.results_file, dpi=75,
                bbox_inches='tight',
                pad_inches=0.1)

    print(">>> Finished generating plots - file saved:", args.results_file)


# execute only if run as a script
if __name__ == "__main__":
    main()
