# coding: utf-8

# ===================================================================
# Mask fMRIs
# Romuald Menuet - June 2018
# ===================================================================
# Summary: This script masks fMRIs whose path is stored as a column
#          of a dataframe to a common affine
# ===================================================================

# 3rd party modules
import argparse
from os import stat
import pickle
import json
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn.input_data import NiftiMasker

# Custom modules
from meta_fmri.preprocess.mask import mask_batch


# =========================
# === MASKING FUNCTIONS ===
# =========================
def mask_mini_batch(fmris, masker, verbose=False):
    if verbose:
        print("  > masking started (one mini-batch, {} fMRIS)"
              .format(len(fmris)))

    X = np.zeros((len(fmris), masker.mask_img.get_data().sum()))
    i = 0
    for idx, fmri in fmris.iteritems():
        X[i] = masker.transform(fmri)
        i += 1

    if verbose:
        print("  > masking ended (one mini-batch)")

    return X


def mask_batch(fmris_file, masker, n_jobs=1, verbose=False):
    fmris = pd.read_csv(fmris_file, index_col=0, header=0,
                        low_memory=False, squeeze=True)

    if verbose:
        print("> File read, {} fMRIs will be  masked".format(len(fmris)))

    fmri_split = np.array_split(fmris, n_jobs)
    ma = lambda x: mask_mini_batch(x, masker, verbose=verbose)
    results = (Parallel(n_jobs=n_jobs, verbose=1, backend="threading")
               (delayed(ma)(x) for x in fmri_split))

    X = np.vstack(results)

    return X


def prepare_mask(global_config=None, n_jobs=1, verbose=False):
    # --------------
    # --- CONFIG ---
    # --------------
    config = global_config["mask"]

    # ---------------
    # --- MASKING ---
    # ---------------
    if verbose:
        print("=" * 30)
        print(" > Masking fMRIs using",
              global_config["mask_file"],
              "as the mask")

    mask = nib.load(global_config["mask_file"])

    masker = NiftiMasker(mask_img=mask).fit()

    X = mask_batch(config["input_file"],
                   masker,
                   n_jobs=n_jobs,
                   verbose=verbose)

    with open(config["output_file"], 'wb') as f:
        pickle.dump(X, f, protocol=4)


# execute only if run as a script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A masking scriptfor data fetched from Neurovault.",
        epilog='''Example: python a4_mask.py -C config.json -j 8 -v'''
    )
    parser.add_argument("-C", "--configuration",
                        default="./config.json",
                        help="Path of the JSON configuration file")
    parser.add_argument("-j", "--jobs",
                        type=int,
                        default=1,
                        help="Number of jobs")
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Activates (many) debugging outputs")

    args = parser.parse_args()

    with open(args.configuration) as f:
        global_config = json.load(f)

    prepare_mask(global_config, args.jobs, args.verbose)
