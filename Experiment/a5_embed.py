# coding: utf-8

# ===================================================================
# Embed fMRIs using a dictionary
# Romuald Menuet - May 2019
# ===================================================================
# Summary: This script embeds masked fMRIs using the provided dictionary
#          and standardize them if required
# ===================================================================

# 3rd party modules
from os import stat
import argparse
import pickle
import json
import numpy as np
import nibabel as nib
from nilearn.input_data import NiftiMasker
from sklearn import preprocessing
import torch
from torch.utils.data.sampler import Sampler, BatchSampler
from meta_fmri.decode.decode import DatasetFromNp


# ===========================
# === EMBEDDING FUNCTIONS ===
# ===========================
def embed_from_atlas(fmris_masked_file, atlas_masked,
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
    with open(fmris_masked_file, 'rb') as f:
        if verbose:
            print("> Loading data to embed...")

        fmris_data = pickle.load(f)

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


def prepare_embed(global_config=None, verbose=False):
    config = global_config["embed"]

    mask = nib.load(global_config["mask_file"])
    atlas = nib.load(global_config["dict_file"])

    masker = NiftiMasker(mask_img=mask).fit()
    atlas_masked = masker.transform(atlas)

    fmris_embedded = embed_from_atlas(
        config["input_file"],
        atlas_masked,
        center=config["center"],
        scale=config["scale"],
        nan_max=config["nan_max"],
        verbose=verbose
    )

    if args.verbose:
        print("> Embedding finished, saving to file...")

    with open(config["output_file"], 'wb') as f:
        pickle.dump(fmris_embedded, f, protocol=4)


# execute only if run as a script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="An embedding script for data fetched from Neurovault "
                    "(resampled and masked).",
        epilog='''Example: python a5_embed.py -C config.json -v'''
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

    prepare_embed(global_config, args.verbose)
