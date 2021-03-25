# IMPORTS
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map, plot_glass_brain


def one_compact_line(string):
    """
    Replaces all multiples spaces/tabs/line-returns by a single space
    to return a compact line from a multi-lines one
    :param string: string
    """
    return re.sub(r"[\s]+", " ", str(string)).strip()


def highly_corr_cols(df, thresh=1.0,
                     return_indices=False,
                     verbose=False):
    df_corr = df.corr()
    to_keep = []
    for i, label in enumerate(df.columns):
        strong_correlates = np.where(np.abs(df_corr.values[i, :i]) >= thresh)[0]
        if len(strong_correlates):
            if verbose:
                print(df.columns[i],
                      "is strongly correlated to",
                      list(df.columns[strong_correlates]),
                      "-> removing it")
        else:
            to_keep += [i]

    if return_indices:
        return to_keep
    else:
        return df.iloc[:, to_keep]


def highly_corr_cols_np(np_array, cols, thresh=1.0,
                        return_indices=False,
                        verbose=False):
    df = pd.DataFrame(np_array, columns=cols)
    if return_indices:
        return highly_corr_cols(df, thresh, return_indices, verbose)
    else:
        return highly_corr_cols(df, thresh, return_indices, verbose).values


def gridsearch_complexity(param_grid):
    """
    Given a parameters grid (list of dict)intended for a grid search,
    provides the number of parameters combination to be explored
    """
    n_combinations = 0
    if isinstance(param_grid, list):
        for param_set in param_grid:
            n_combinations_set = 1
            for key in param_set:
                n_combinations_set *= len(param_set[key])
            n_combinations += n_combinations_set
    else:
        n_combinations_set = 1
        for key in param_grid:
            n_combinations_set *= len(param_grid[key])
        n_combinations += n_combinations_set

    return n_combinations


def yes_or_no(question):
    answer = input(question + "(y/n): ").lower().strip()
    print("")
    while not(answer == "y" or answer == "yes"
              or answer == "n" or answer == "no"):
        print("Input yes or no")
        answer = input(question + "(y/n):").lower().strip()
        print("")
    if answer[0] == "y":
        return True
    else:
        return False


def mask_rows(mask, *args):
    return (arg[mask] for arg in args)


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


def plot_stack_embedded(x,
                        atlas_masked_list,
                        masker,
                        plot_type="glass_brain",
                        title="",
                        axes=None):
    total_res = 0
    for atlas_masked in atlas_masked_list:
        total_res += len(atlas_masked)
    assert len(x) == total_res, "dimensions issues"

    masked_data = []
    offset = 0
    for atlas_masked in atlas_masked_list:
        masked_data += [
            x[offset:offset+len(atlas_masked)] @ atlas_masked
        ]
        offset += len(atlas_masked)
    vox = masker.inverse_transform(np.sum(masked_data, axis=0))
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
