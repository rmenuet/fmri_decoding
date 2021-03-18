# coding: utf-8

# ===================================================================
# Generates plots illustrating a decoding model performance
# Romuald Menuet - May 2019
# ===================================================================
# Summary: Scripts that generates a performance plot
#          as in TODO:ajouter_nom_article
# ===================================================================

import argparse
import json
import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from nilearn.plotting import plot_stat_map, plot_glass_brain
from nilearn.input_data import NiftiMasker

from b_decoding_experiment import PytorchEstimator


def plot_embedded(x,
                  atlas_masked,
                  masker,
                  plot_type="glass_brain",
                  title="",
                  axes=None):
    """
    Plots the means of statistical maps embedded using an atlas.

    :param x: numpy.ndarray (n_atlas_components)
              The array of the components from an embedded statistical map.
    :param atlas_masked:
    :param masker:
    :param plot_type:
    :param title:
    :param axes:
    :return:
    """
    mask = x @ atlas_masked
    vox = masker.inverse_transform(mask)
    if plot_type == "glass_brain":
        return plot_glass_brain(
            vox,
            display_mode='xz',
            plot_abs=False,
            threshold=None,
            # colorbar=True,
            cmap=plt.cm.bwr,
            title=title,
            axes=axes
        )
    else:
        return plot_stat_map(
            vox,
            threshold=None,
            colorbar=True,
            cmap=plt.cm.bwr,
            title=title,
            axes=axes
        )




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
    parser.add_argument("-r1", "--result_file",
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
    # parser.add_argument("-e", "--encodings",
    #                     default="../Data/models/encodings.p",
    #                     help="Path of the pickle dump of the dictionary "
    #                          "of encoding maps")
    parser.add_argument("-c", "--concepts",
                        default="../Data/concepts.csv",
                        help="Path of the CSV file of comma separated concepts"
                             ", ordered as the columns of the labels file")
    parser.add_argument("-r2", "--results_file",
                        default="../Data/results/maps.PNG",
                        help="Path to the file where results are saved")

    args = parser.parse_args()

    # Load Cognitive Atlas categories
    with open("Data/labels/cogatlas_concepts_categories_mapping.json", "r") as file:
        concept_to_category = json.load(file)

    # --------------------
    # --- DATA LOADING ---
    # --------------------
    results_model = pd.read_csv(args.perf_file, index_col=0)
    results_model = (
        results_model
        .assign(category=lambda df: df.index.map(concept_to_category).fillna("Unknown"))
        .sort_values(["category", "AUC TEST"], ascending=[True, False])
    )
    results_model.index.name = "Concept"

    categories_lengths = dict(
        results_model.category.value_counts()
    )
    total_number_of_concepts = results_model.category.count()
    # ------------------------
    # --- PLOT PERFORMANCE ---
    # ------------------------
    sns.set(style="whitegrid")
    sns.set_context("notebook", font_scale=1.1)
    font = {'fontname': 'DejaVu Sans Mono'}
    ylabels = list(results_model.index)
    colors_mapping = {
        "Action": [rgb / 255. for rgb in [238, 181, 235]],
        "Attention": [rgb / 255. for rgb in [194, 109, 188]],
        "Emotion": [rgb / 255. for rgb in [200, 244, 249]],
        'Executive/Cognitive Control': [rgb / 255. for rgb in [60, 172, 174]],
        'Language': [rgb / 255. for rgb in [250, 190, 192]],
        'Learning and Memory': [rgb / 255. for rgb in [248, 92, 112]],
        'Perception': [rgb / 255. for rgb in [243, 121, 112]],
        'Reasoning and Decision Making': [rgb / 255. for rgb in [228, 61, 64]],
        'Social Function': "brown",
        'Unknown': "olive",
    }
    colors = list(results_model.category.map(colors_mapping))

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
                 color=colors)

    y_min, y_max = ax_twin.get_ylim()
    y_span = y_max - y_min
    x_max, x_min = ax_twin.get_xlim()
    # current_y_offset = y_max
    current_y_offset = 1.  # - 0.5 / (total_number_of_concepts + 1)

    linewidth = 15
    horizontal_offset = 0.05
    vertical_offset = 0.75
    for index, category in enumerate(sorted(categories_lengths.keys())):
        y_top = current_y_offset - 0.5 / (total_number_of_concepts + 1)
        y_bottom = y_top - categories_lengths[category] / (total_number_of_concepts + 1)
        ax_twin.axvline(
            x_max - horizontal_offset,
            y_top - vertical_offset / (total_number_of_concepts + 1),
            y_bottom + vertical_offset / (total_number_of_concepts + 1),
            color=colors_mapping[category],
            linewidth=linewidth,
        )
        ax_twin.text(
            x_min - horizontal_offset,
            (y_bottom + y_top) / 2,
            (category.replace(" ", "\n").replace("/", "\n") if categories_lengths[category] < 8 else category),
            ha="center",
            # va="center",
            rotation="vertical",
            fontsize=8,
            transform=ax_twin.transAxes,
        )
        current_y_offset = y_bottom + 0.5 / (total_number_of_concepts + 1)
        # break

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

    mask = nib.load(args.mask)
    masker = NiftiMasker(mask_img=mask).fit()

    atlas = nib.load(args.atlas)
    atlas_masked = masker.transform(atlas)

    # with open(args.encodings, 'rb') as f:
    #     encodings_dict = pickle.load(f)

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
        # pl_enc = plot_embedded(
        #     encodings_dict[concept].flatten(),
        #     atlas_masked, masker,
        #     plot_type="glass_brain", axes=enc_axes[i]
        # )
        # pl_enc.title("Enc.", color='k', bgcolor='w', alpha=0, size=16,
        #              weight='bold')

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
