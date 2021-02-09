# IMPORTS
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map, plot_glass_brain


# GENERIC HELPER FUNCTIONS
from sklearn.metrics import roc_auc_score


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


def mkdir(path):
    """Creates a folder if it does not already exist,
    returns True if the folder was created, else False."""
    try:
        created = False
        if not os.path.exists(path):
            os.makedirs(path)
            created = True
        return created
    except OSError:
        print('Error: Creating directory. ' + path)


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


def recall_n(y_pred, y_truth,
             n=10, thresh=0.95, reduce_mean=False):
    """

    :y_pred:
    :y_truth:
    :n:
    :thresh:
    :reduce_mean:
    """

    assert (y_pred.ndim in (1, 2)) and (y_truth.ndim in (1, 2)), "arrays should be of dimension 1 or 2"
    assert y_pred.shape == y_truth.shape, "both arrays should have the same shape"

    if y_pred.ndim == 1:
        # recall@n for a single sample
        targets = np.where(y_truth >= thresh)[0]
        pred_n_first = np.argsort(y_pred)[::-1][:n]

        if len(targets) > 0:
            ratio_in_n = len(np.intersect1d(targets,
                                            pred_n_first)) / len(targets)
        else:
            ratio_in_n = np.nan

        return ratio_in_n
    else:
        # recall@n for a dataset (mean of recall@n for all samples)
        result = np.zeros(len(y_pred))
        for i, (sample_y_pred, sample_y_truth) in enumerate(zip(y_pred, y_truth)):
            result[i] = recall_n(sample_y_pred, sample_y_truth,
                                 n, thresh)
        if reduce_mean:
            return np.nanmean(result)
        else:
            return result


def mean_rank(y_pred, y_truth,
              thresh=0.95, reduce_mean=False):
    if y_pred.ndim == 1:
        # mean rank for a single sample
        targets = np.where(y_truth >= thresh)[0]
        pred_ordered = np.argsort(y_pred)[::-1]
        result = np.mean(np.argwhere(np.isin(pred_ordered, targets)))
        return result

    else:
        # mean rank for a dataset (mean of mean ranks for all samples)
        result = np.zeros(len(y_pred))
        for i, (sample_y_pred, sample_y_truth) in enumerate(zip(y_pred, y_truth)):
            result[i] = mean_rank(sample_y_pred,
                                  sample_y_truth,
                                  thresh)
        if reduce_mean:
            return result.mean()
        else:
            return result


def mean_auc(y_pred, y_truth):
    n_labels_to_find = 0
    auc_tot = 0
    for i in range(y_truth.shape[1]):
        if (y_truth[:, i].sum()) and (0 in y_truth[:, i]):
            n_labels_to_find += 1
            auc_tot += roc_auc_score(y_truth[:, i], y_pred[:, i])

    assert n_labels_to_find, "No label found"

    return auc_tot / n_labels_to_find


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
