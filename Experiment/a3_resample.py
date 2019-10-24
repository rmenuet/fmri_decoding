# coding: utf-8

# ===================================================================
# Resample fMRIs to common affine
# Romuald Menuet - June 2018
# ===================================================================
# Summary: This script resamples fMRIs whose path is stored as a column
#          of a dataframe to a common affine
# ===================================================================

# 3rd party modules
import argparse
import os
import json
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img


# ============================
# === RESAMPLING FUNCTIONS ===
# ============================
def resample_mini_batch(fmris, ref_map, cache_folder,
                        overwrite=False,
                        file_field="absolute_path",
                        suffix="_resamp",
                        interpolation="linear",
                        verbose=False):
    """
    Batch resampling of fMRIs whose filenames are stored in a Pandas DataFrame
    column, and that are stored as gzip ('*.gz') compressed Nifti  files
    (useful to resample files downloaded from Neurovault).

    :param fmris: pandas.DataFrame
        The dataframe where the filenames of fMRIs are stored.
    :param cache_folder: string
        The folder where to store the new files.
    :param ref_map: string
        The absolute path of a reference Nifti file whose affine will be used to
        resample others.
    :param overwrite: boolean
        If True, existing - allready resampled - files will be overwritten by
        the new ones.
    :param file_field: string
        The name of the columnn where the absolute path of the files to resample
        is stored.
    :param suffix: string
        The suffix to apply to both filenames and the column where to store
        their path.
    :param interpolation: string
        The interpolation (as named in nilearn's resample_to_img function) to
        apply.
    :param verbose: boolean
        Whether to print debugging messages.
    :return: pandas.DataFrame, int, set of indexes
        - The dataframe with the resampled absolute pathes
        - The number of encountered errors as well
        - The indexes of the fMRIs where they occured.
    """
    fmris_resamp = fmris

    errors = 0
    failed_MRIs = set()
    resamp_field = file_field + suffix
    if resamp_field not in fmris_resamp.columns:
        fmris_resamp[resamp_field] = ""

    for idx, row in fmris.iterrows():
        file = row[file_field]
        if overwrite or not os.path.isfile(row[resamp_field]):
            try:
                # new file path by simply removing the ".gz" at the end
                new_file = (cache_folder
                            + "/fmri_"
                            + str(idx)
                            + suffix
                            + ".nii.gz")
                map_orig = nib.load(file)
                map_resampled = resample_to_img(map_orig,
                                                ref_map,
                                                interpolation=interpolation)
                map_resampled.to_filename(new_file)
                fmris_resamp.at[idx, resamp_field] = new_file
                if verbose:
                    print(file, "resampled to", new_file)
            except Exception as e:
                print("Error resampling fMRI", idx)
                print(str(e))
                errors += 1
                failed_MRIs.update((idx,))
                # break

    return fmris_resamp, errors, failed_MRIs


def resample_batch(fmris_file,
                   ref_file,
                   cache_folder, path_file,
                   overwrite=False,
                   file_field="absolute_path",
                   interpolation="linear",
                   suffix="_resamp",
                   n_jobs=4,
                   verbose=False):

    fmris = pd.read_csv(fmris_file, low_memory=False, index_col=0)

    if verbose:
        print("> File read,",
              len(fmris[fmris["kept"]]),
              "fMRIs will be resampled with",
              interpolation,
              "interpolation")

    ref_map = nib.load(ref_file)

    fmris_split = np.array_split(fmris[fmris["kept"]], n_jobs)

    resamp = lambda x: resample_mini_batch(x, ref_map, cache_folder, overwrite,
                                           file_field=file_field,
                                           interpolation=interpolation,
                                           suffix=suffix,
                                           verbose=verbose)
    results = (Parallel(n_jobs=n_jobs, verbose=1, backend="threading")
               (delayed(resamp)(x) for x in fmris_split))

    fmris_resamp = pd.DataFrame()
    errors = 0
    failed = set()
    for result in results:
        fmris_resamp = pd.concat([fmris_resamp, result[0]])
        errors += result[1]
        failed.update(result[2])

    pd.Series(list(failed)).to_csv(cache_folder + "/failed_resamples.csv",
                                   header=True)

    fmris_resamp[file_field + suffix].to_csv(path_file, header=True)

    return fmris_resamp


def prepare_resample(global_config=None, n_jobs=1, verbose=False):
    # --------------
    # --- CONFIG ---
    # --------------
    config          = global_config["resample"]
    meta_path       = global_config["meta_path"]
    cache_path      = global_config["cache_path"]
    fmris_meta_file = meta_path + global_config["meta_file"]
    target_affine   = global_config["dict_file"]
    path_file       = cache_path + config["output_file"]


    # ------------------
    # --- RESAMPLING ---
    # ------------------
    if verbose:
        print("=" * 30)
        print(" > Resampling fMRIs using",
              target_affine,
              "as the target affine")

    resample_batch(fmris_meta_file,
                   target_affine,
                   cache_path, path_file,
                   overwrite=config["overwrite"],
                   file_field=config["input_field"],
                   interpolation=config["interpolation"],
                   n_jobs=n_jobs,
                   verbose=verbose)

    print(">>> Resampling done")


# execute only if run as a script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A resampling script for data fetched from Neurovault.",
        epilog='''Example: python a3_resample.py -C config.json -j 8 -v'''
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
                        help="Activates (many) outputs")

    args = parser.parse_args()

    with open(args.configuration) as f:
        global_config = json.load(f)

    prepare_resample(global_config, args.jobs, args.verbose)
