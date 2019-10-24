# coding: utf-8

# ===================================================================
# Impute missing values of embedded fMRIs
# Romuald Menuet - May 2019
# ===================================================================
# Summary: This script imputes missing values of embedded fMRIs
#          by using the given strategy (median or sampling imputation)
#
# Note:    We found that this preprocessing did neither increase or
#          decrease performance at the time of our experiments
#          but it is still usefull to prevent models to overfit
#          fMRIs cropping specific to a study that provides most
#          samples of a given label: this yields more sensible
#          encoding and decoding maps
# ===================================================================

# ========================
# === IMPORTS & CONFIG ===
# ========================
# 3rd party modules
import argparse
import pickle
import json
import numpy as np
from sklearn import preprocessing


# ============================
# === IMPUTATION FUNCTIONS ===
# ============================
def impute_samples(X):
    """
    Replaces NaN and 0 values in a 2D numpy array by samples from the same
    column
    """
    # TODO: switch to SKL SampleImputer once released
    X_result = X.copy()

    X_result[X_result == 0] = np.nan

    for i in range(X_result.shape[1]):
        missing = ~np.isfinite(X_result[:, i])
        n_missing = missing.sum()
        if n_missing:
            X_result[missing, i] = np.random.choice(X_result[~missing, i],
                                                    size=n_missing,
                                                    replace=True)

    return X_result


def impute(file,
           strategy='median',
           scale=False):
    with open(file, 'rb') as f:
        data = pickle.load(f)

    if (strategy == 'mean') or (strategy == 'median'):
        imputer = preprocessing.Imputer(strategy=strategy)
        result = imputer.fit_transform(data)
    else:
        result = impute_samples(data)

    if scale:
        result = preprocessing.scale(result,
                                     with_mean=False,
                                     with_std=scale,
                                     axis=1)

    return result


def prepare_impute(global_config=None):
    config = global_config["impute"]

    imputed_values = impute(config["input_file"],
                            config["imputation"],
                            config["scale"])

    with open(config["output_file"], 'wb') as f:
        pickle.dump(imputed_values, f, protocol=4)


# execute only if run as a script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="An imputation for missing components of embedded fMRIs"
                    " fetched from Neurovault.",
        epilog='''Example: python a6_impute.py -C config.json -v'''
    )
    parser.add_argument("-C", "--configuration",
                        default="./config.json",
                        help="Path of the JSON configuration file")
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Activates (many) debugging outputs")

    args = parser.parse_args()

    with open(args.configuration) as f:
        global_config = json.load(f)

    prepare_impute(global_config)
