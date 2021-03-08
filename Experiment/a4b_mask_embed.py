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
import os
import pickle
import json

import pandas as pd
import numpy as np
import nibabel as nib
from joblib import Parallel, delayed
from nilearn.input_data import NiftiMasker
from sklearn import preprocessing

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
    if verbose:
        print("> Start masking...")

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

    if verbose:
        print("> Start fitting mask...")

    masker = NiftiMasker(mask_img=mask).fit()

    if verbose:
        print("> Applying fitted mask...")

    fmris_masked = mask_batch(
        os.path.join(global_config["cache_path"], config["input_file"]),
        masker,
        n_jobs=n_jobs,
        verbose=verbose
    )

    return fmris_masked


# ===========================
# === EMBEDDING FUNCTIONS ===
# ===========================
def embed_from_atlas(fmris_masked, atlas_masked,
                     center=False, scale=False, absolute=False,
                     nan_max=1.0, tol=0.1,
                     projection=False,
                     verbose=False):
    """
    Embeds fMRI stat-maps using a dictionary of components,
    either projecting on it or regressing the data over it.

    :param fmris_masked_file:
    :param atlas_masked:
    :param center:
    :param scale:
    :param absolute:
    :param nan_max:
    :param projection:
    :param verbose:
    :return:
    """
    fmris_data = fmris_masked

    if absolute:
        if verbose:
            print("> Taking positive part...")
        fmris_data[fmris_data < 0] = 0

    if center or scale:
        if verbose:
            print("> Scaling...")
        fmris_data = preprocessing.scale(fmris_data,
                                         with_mean=center,
                                         with_std=scale,
                                         axis=1)

    if verbose:
        print("Calculating components...")

    if projection:
        result = fmris_data @ atlas_masked.T
    else:
        result = fmris_data @ np.linalg.pinv(atlas_masked)

    if nan_max < 1.0:
        if verbose:
            print("> Setting components with too many missing voxels to NaN...")
            print("  > Treshold =", nan_max)

        mask_missing_voxels = (np.abs(fmris_data) < tol)

        # if a voxel is missing and is part of a component,
        # this component is set to NaN:
        mask_result_nans = (mask_missing_voxels @ atlas_masked.T) > nan_max
        result[mask_result_nans] = np.nan

        if verbose:
            print("  > Total number of missing voxels:",
                  mask_missing_voxels.sum())
            print("  > Total number of missing components:",
                  mask_result_nans.sum())

    return result


def prepare_embed(fmris_masked, global_config=None, verbose=False):
    if verbose:
        print("> Start embedding...")

    config = global_config["embed"]

    mask = nib.load(global_config["mask_file"])
    atlas = nib.load(global_config["dict_file"])

    masker = NiftiMasker(mask_img=mask).fit()
    atlas_masked = masker.transform(atlas)

    fmris_embedded = embed_from_atlas(
        fmris_masked,
        atlas_masked,
        center=config["center"],
        scale=config["scale"],
        nan_max=config["nan_max"],
        verbose=verbose
    )

    if verbose:
        print("> Embedding finished, saving to file...")

    with open(config["output_file"], 'wb') as f:
        pickle.dump(fmris_embedded, f, protocol=4)


# execute only if run as a script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A masking scriptfor data fetched from Neurovault.",
        epilog='''Example: python a4_mask.py -C config.json -j 8 -v'''
    )
    parser.add_argument("-C", "--configuration",
                        default="./preparation_config.json",
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

    fmris_masked = prepare_mask(global_config, args.jobs, args.verbose)
    prepare_embed(fmris_masked, global_config, args.verbose)
