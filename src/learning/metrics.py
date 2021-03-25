import numpy as np
from sklearn.metrics import roc_auc_score


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
