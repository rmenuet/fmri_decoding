# coding: utf-8

# ===================================================================
# Filter fMRIs & collections based upon config file
# Romuald Menuet - June 2018, updated January 2019
# ===================================================================
# Summary: This script enriches the metadata of fMRIs collections
#          and then filters them (only keeps those that seem relevant)
#          based on a JSON configuration file with filtering criteria
# ===================================================================

# 3rd party modules
import argparse
import json
from functools import partial
from os import stat
from pprint import pprint

import numpy as np
import pandas as pd
from nibabel import load as nib_load
from joblib import Parallel, delayed


# ================================
# === ENRICH METADATA & FILTER ===
# ================================
def extract_meta(fmris_meta, config):
    """
    Extract additional metadata from fMRIs listed in a dataframe by analyzing
    the stat-maps
    Warning: Since this function loads all stat-maps in memory and analyze
             their values, it can be quite long for thousands of fMRIs. It is
             one of the main bottlenecks of this fetching and filtering pipeline
             and should be parallelized if possible.

    :param fmris_meta: pandas.DataFrame
        Dataframe with the metadata loaded from the sources (Neurovault, HCP...)
    :param config: dict
        Description of some of the data to extract

    :return: pandas.DataFrame
        Input dataframe with additional metadata.
    """
    for idx, row in fmris_meta.iterrows():
        try:
            fmri = nib_load(row["absolute_path"])
        except BaseException:
            print("Error with", row["absolute_path"])
            raise RuntimeError("Error unzipping a file")

        res_x, res_y, res_z = fmri.header.get_zooms()
        dim_x, dim_y, dim_z = fmri.header.get_data_shape()
        fmris_meta.at[idx, "res_x"] = res_x
        fmris_meta.at[idx, "res_y"] = res_y
        fmris_meta.at[idx, "res_z"] = res_z
        fmris_meta.at[idx, "dim_x"] = dim_x
        fmris_meta.at[idx, "dim_y"] = dim_y
        fmris_meta.at[idx, "dim_z"] = dim_z

        try:
            mat = fmri.get_data().astype(float)
        except BaseException:
            print("Error with", row["absolute_path"])
            raise RuntimeError("Error unzipping a file")

        # mat = np.nan_to_num(mat)
        mat[mat == np.inf] = np.nan
        mat[mat == -np.inf] = np.nan
        fmris_meta.at[idx, "n_values"] = len(np.unique(mat[~np.isnan(mat)]))
        fmris_meta.at[idx, "min_value"] = np.nanmin(mat)
        fmris_meta.at[idx, "max_value"] = np.nanmax(mat)
        try:
            fmris_meta.at[idx, "min_pos_value"] = mat[mat > 0].nanmin()
        except BaseException:
            fmris_meta.at[idx, "min_pos_value"] = 0
        try:
            fmris_meta.at[idx, "max_neg_value"] = mat[mat < 0].nanmax()
        except BaseException:
            fmris_meta.at[idx, "max_neg_value"] = 0

        if mat[(~np.isnan(mat)) & (mat != 0)].any():
            fmris_meta.at[idx, "first_quantile"] = np.percentile(
                mat[(~np.isnan(mat)) & (mat != 0)],
                config["centered_param"]
            )
            fmris_meta.at[idx, "last_quantile"] = np.percentile(
                mat[(~np.isnan(mat)) & (mat != 0)],
                100 - config["centered_param"]
            )
        else:
            fmris_meta.at[idx, "first_quantile"] = 0
            fmris_meta.at[idx, "last_quantile"] = 0

        fmris_meta.at[idx, "hash"] = hash(mat[~np.isnan(mat)].tostring())
        fmri.uncache()

    return fmris_meta


def filter_data(colls_file,
                fmris_file,
                fmris_meta_file,
                config, n_jobs,
                verbose=False):
    """
    Extract data from fMRIs and perform filtering based upon the provided
    configuration.

    :param colls_file: str
        CSV file  where collections metadata were previously saved.
    :param fmris_file: str
        CSV file  where fMRIs metadata (from source) were previously saved.
    :param fmris_meta_file: str
        CSV file  where fMRIs metadata (computed by this function) are saved.
    :param config: dict
        Description of some of the data to extract
    :param n_jobs: int
        Number of threads used by joblib to perform stat-maps analysis.
    :param verbose: bool
        Activates verbose mode for troubleshooting.

    :return: (pandas.DataFrame, pandas.DataFrame, int, int)
        Dataframe with collections metadata,
        Dataframe with fMRIs metadata,
        Number of collections where at least 1 fMRI was kept,
        Number of kept fMRIs
    """

    # Load metadata obtained from sources
    colls = pd.read_csv(colls_file, low_memory=False, index_col=0)
    fmris = pd.read_csv(fmris_file, low_memory=False, index_col=0)

    if verbose:
        print(">>> Starting filtering from {} fMRIs",
              len(fmris))

    # Nifty files metadata extraction
    if verbose:
        print("... extraction from Nifty files ...")

    fmris_meta_split = np.array_split(fmris, n_jobs * 3)

    results = (
        Parallel(n_jobs=n_jobs, verbose=1, backend="threading")
        (delayed(partial(extract_meta, config=config))(x) for x in fmris_meta_split)
    )

    fmris_meta = pd.concat(results)

    # Filtering criteria computation
    fmris_meta["has_cog_paradigm"] = ~fmris["cognitive_paradigm_cogatlas"].isnull()
    fmris_meta["has_tags"]         = ~fmris["tags"].isnull()
    fmris_meta["has_task"]         = ~fmris["task"].isnull()
    fmris_meta["has_cont"]         = ~fmris["contrast_definition"].isnull()
    fmris_meta["has_desc"]         = ~fmris["description"].isnull()
    fmris_meta["has_mod"]          = ~fmris["modality"].isnull()
    fmris_meta["not_duplicate"]    = ~fmris_meta.duplicated("hash", keep="last")
    fmris_meta["enough_coverage"]  = fmris["brain_coverage"] > config["enough_coverage_param"]
    fmris_meta["enough_values"]    = fmris_meta["n_values"] >= config["enough_values_param"]
    fmris_meta["centered"]         = (  (fmris_meta["first_quantile"] < 0)
                                      & (fmris_meta["last_quantile"]  > 0))
    fmris_meta["unthresholded"]    = (  (fmris_meta["min_pos_value"] - fmris_meta["max_neg_value"])
                                      / (fmris_meta["max_value"]     - fmris_meta["min_value"])
                                      < config["unthresholded_param"])
    fmris_meta["big_enough"]       = ((- fmris_meta["min_value"] > config["min_max_abs_val_param"])
                                      &
                                      (fmris_meta["max_value"] > config["min_max_abs_val_param"]))
    fmris_meta["small_enough"]     = ((- fmris_meta["min_value"] < config["max_max_abs_val_param"])
                                      &
                                      (fmris_meta["max_value"] < config["max_max_abs_val_param"]))
    fmris_meta["proper_mod"] = False
    for col in config["proper_mod_param"]:
        fmris_meta["proper_mod"]   = (  fmris_meta["proper_mod"]
                                      | (fmris["modality"] == col))
    fmris_meta["proper_type"] = False
    for col in config["proper_type_param"]:
        fmris_meta["proper_type"]  = (  fmris_meta["proper_type"]
                                      | (fmris["map_type"] == col))

    # Use of filtering criteria to build summary filtering column
    fmris_meta["kept"] = (
        (fmris_meta["has_cog_paradigm"]   | (not config["has_cog_paradigm"]))
        & (fmris_meta["has_tags"]         | (not config["has_tags"]))
        & (fmris_meta["has_task"]         | (not config["has_task"]))
        & (fmris_meta["has_cont"]         | (not config["has_cont"]))
        & (fmris_meta["has_desc"]         | (not config["has_desc"]))
        & (fmris_meta["has_mod"]          | (not config["has_mod"]))
        & (fmris_meta["proper_mod"]       | (not config["proper_mod"]))
        & (fmris_meta["not_duplicate"]    | (not config["not_duplicate"]))
        & (fmris_meta["enough_coverage"]  | (not config["enough_coverage"]))
        & (fmris_meta["enough_values"]    | (not config["enough_values"]))
        & (fmris_meta["centered"]         | (not config["centered"]))
        & (fmris_meta["unthresholded"]    | (not config["unthresholded"]))
        & (fmris_meta["big_enough"]       | (not config["min_max_abs_val"]))
        & (fmris_meta["small_enough"]     | (not config["max_max_abs_val"]))
        & (fmris_meta["proper_type"]      | (not config["proper_type"]))
    )

    fmris_all = fmris.join(fmris_meta.drop("absolute_path", axis=1),
                           lsuffix='_orig')

    if verbose:
        print("-"*30)
        print(">>> Results of the filtering:")
        print("  > Number of fMRIs: with a 'CogAtlas paradigm':",
              fmris_meta["has_cog_paradigm"].sum())
        print("                     with 'tags':",
              fmris_meta["has_tags"].sum())
        print("                     with a 'task':",
              fmris_meta["has_task"].sum())
        print("                     with a 'description':",
              fmris_meta["has_desc"].sum())
        print("                     with a 'contrast_definition':",
              fmris_meta["has_cont"].sum())
        print("                     with a 'modality':",
              fmris_meta["has_mod"].sum())
        print("                     with the proper 'modality':",
              fmris_meta["proper_mod"].sum())
        print("                     that are not duplicates from older ones:",
              fmris_meta["not_duplicate"].sum())
        print("                     with more than 65% brain coverage:",
              fmris_meta["enough_coverage"].sum())
        print("                     with 10 or more values:",
              fmris_meta["enough_values"].sum())
        print("                     whose values are (somewhat) centered:",
              fmris_meta["centered"].sum())
        print("                     that are not thresholded:",
              fmris_meta["unthresholded"].sum())
        print("                     that have big enough values:",
              fmris_meta["big_enough"].sum())
        print("                     that have small enough values:",
              fmris_meta["small_enough"].sum())
        print("                     of the proper map type:",
              fmris_meta["proper_type"].sum())
        print("-"*30)

    # === per collection consolidation ===

    collection_grouped_fmris_all = fmris_all.groupby("collection_id")

    has_cog_paradigm = fmris_all.groupby("collection_id")["has_cog_paradigm"].sum()
    has_cog_paradigm = has_cog_paradigm.astype(int)

    has_tags = collection_grouped_fmris_all["has_tags"].sum()
    has_tags = has_tags.astype(int)

    has_task = collection_grouped_fmris_all["has_task"].sum()
    has_task = has_task.astype(int)

    has_cont = collection_grouped_fmris_all["has_cont"].sum()
    has_cont = has_cont.astype(int)

    has_desc = collection_grouped_fmris_all["has_desc"].sum()
    has_desc = has_desc.astype(int)

    has_mod = collection_grouped_fmris_all["has_mod"].sum()
    has_mod = has_mod.astype(int)

    proper_mod = collection_grouped_fmris_all["proper_mod"].sum()
    proper_mod = proper_mod.astype(int)

    not_duplicate = collection_grouped_fmris_all["not_duplicate"].sum()
    not_duplicate = not_duplicate.astype(int)

    enough_coverage = collection_grouped_fmris_all["enough_coverage"].sum()
    enough_coverage = enough_coverage.astype(int)

    enough_values = collection_grouped_fmris_all["enough_values"].sum()
    enough_values = enough_values.astype(int)

    centered = collection_grouped_fmris_all["centered"].sum()
    centered = centered.astype(int)

    unthresholded = collection_grouped_fmris_all["unthresholded"].sum()
    unthresholded = unthresholded.astype(int)

    big_enough = collection_grouped_fmris_all["big_enough"].sum()
    big_enough = big_enough.astype(int)
    small_enough = collection_grouped_fmris_all["small_enough"].sum()
    small_enough = small_enough.astype(int)

    proper_type = collection_grouped_fmris_all["proper_type"].sum()
    proper_type = proper_type.astype(int)

    kept = collection_grouped_fmris_all["kept"].sum()
    kept = kept.astype(int)

    colls = (colls.join(has_cog_paradigm)
                  .join(not_duplicate)
                  .join(enough_coverage)
                  .join(enough_values)
                  .join(centered)
                  .join(unthresholded)
                  .join(big_enough)
                  .join(small_enough)
                  .join(proper_type)
                  .join(has_tags)
                  .join(has_task)
                  .join(has_cont)
                  .join(has_desc)
                  .join(has_mod)
                  .join(proper_mod)
                  .join(kept))

    if config["not_temporary"]:
        temporary = colls["name"].str.contains("temporary collection", na=False)
        if verbose:
            print(">>> Num. of collections excluded because they are"
                  " temporary:",
                  temporary.sum())

        colls.loc[temporary, "kept"] = 0

    if config["min_fmri"] > 1:
        too_small = colls["kept"] < config["min_fmri"]
        if verbose:
            print(">>> Num. of collections excluded because of too few fMRIs:",
                  too_small.sum())

        colls.loc[too_small, "kept"] = 0

    # TODO : finir

    colls["ratio_kept"] = (  colls["kept"]
                           / colls["downloaded_images"])

    n_kept_colls = colls[colls["kept"] > 0].shape[0]
    n_kept_fmris = colls["kept"].sum()

    if verbose:
        print(">>> Number of (partially) kept collections:",
              n_kept_colls,
              "out of",
              colls.shape[0])
        print(">>> Number of kept fMRIs:",
              n_kept_fmris,
              "out of",
              fmris.shape[0])

    fmris.to_csv(fmris_file, header=True)
    fmris_meta.to_csv(fmris_meta_file, header=True)
    colls.to_csv(colls_file, header=True)

    return colls, fmris_all, n_kept_colls, n_kept_fmris


def prepare_filter(global_config=None, n_jobs=1, verbose=False):
    # --------------
    # --- CONFIG ---
    # --------------
    config     = global_config["filter"]
    meta_path  = global_config["meta_path"]
    colls_file = meta_path + "colls.csv"
    fmris_file = meta_path + "fmris.csv"
    colls_meta_file = meta_path + config["output_colls"]
    fmris_meta_file = meta_path + global_config["meta_file"]

    if verbose:
        print("=" * 30)
        print(" > Used filtering configuration:")
        pprint(config)

    # -----------------
    # --- FILTERING ---
    # -----------------
    colls, fmris, n_kept_colls, n_kept_fmris = filter_data(
        colls_file,
        fmris_file,
        fmris_meta_file,
        config, n_jobs,
        verbose,
    )

    # --------------------------------
    # --- ADDITIONAL METADATA SAVE ---
    # --------------------------------

    collection_grouped_fmris = fmris.groupby("collection_id")

    n_cog_params = (collection_grouped_fmris
                    ["cognitive_paradigm_cogatlas"]
                    .nunique())
    n_cog_params.name = "n_cog_params"
    n_cog_params = n_cog_params.astype(int)
    cog_params = (collection_grouped_fmris
                  ["cognitive_paradigm_cogatlas"]
                  .unique())
    cog_params.name = "cog_params"

    n_contrasts = (collection_grouped_fmris
                   ["contrast_definition"]
                   .nunique())
    n_contrasts.name = "n_contrasts"
    n_contrasts = n_contrasts.astype(int)
    contrasts = (collection_grouped_fmris
                 ["contrast_definition"]
                 .unique())
    contrasts.name = "contrasts"

    colls = (colls.join(n_cog_params)
                  .join(cog_params)
                  .join(n_contrasts)
                  .join(contrasts))

    colls.to_csv(colls_meta_file)

    cognitive_paradigm_grouped_fmris = fmris.groupby("cognitive_paradigm_cogatlas")

    fmris_per_task = (
        cognitive_paradigm_grouped_fmris
        .size()
        .sort_values(ascending=False)
    )

    colls_per_task = (
        cognitive_paradigm_grouped_fmris["collection_id"]
        .nunique()
        .sort_values(ascending=False)
    )

    fmris_kept = fmris[fmris["kept"]]
    cognitive_paradigm_grouped_fmris_kept = fmris_kept.groupby("cognitive_paradigm_cogatlas")

    fmris_per_task_kept = (
        cognitive_paradigm_grouped_fmris_kept
        .size()
        .sort_values(ascending=False)
    )

    colls_per_task_kept = (
        cognitive_paradigm_grouped_fmris_kept["collection_id"]
        .nunique()
        .sort_values(ascending=False)
    )

    max_fmris_per_task = (fmris
                          .groupby(["cognitive_paradigm_cogatlas",
                                    "collection_id"])
                          .size()
                          .groupby(["cognitive_paradigm_cogatlas"])
                          .max()
                          .sort_values(ascending=False))

    max_fmris_per_task_kept = (fmris_kept
                               .groupby(["cognitive_paradigm_cogatlas",
                                         "collection_id"])
                               .size()
                               .groupby(["cognitive_paradigm_cogatlas"])
                               .max()
                               .sort_values(ascending=False))

    paradigms = pd.DataFrame(dict(colls=colls_per_task,
                                  colls_kept=colls_per_task_kept,
                                  fmris=fmris_per_task,
                                  fmris_kept=fmris_per_task_kept,
                                  max_fmris=max_fmris_per_task,
                                  max_fmris_kept=max_fmris_per_task_kept))
    paradigms = paradigms.fillna(0)
    paradigms = paradigms.astype(int)
    paradigms.to_csv(meta_path + "paradigms.csv", header=True)

    if verbose:
        print(">>> Filtering OK, {} kept colls and {} kepts fMRIs"
              .format(n_kept_colls, n_kept_fmris))


# execute only if run as a script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A fetching/preprocessing pipeline for data fetched from"
                    " Neurovault.",
        epilog='''Example: python a2_filter.py -v -j 4'''
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
                        help="Activates (many) outputs")

    args = parser.parse_args()

    with open(args.configuration) as f:
        global_config = json.load(f)

    prepare_filter(global_config, args.jobs, args.verbose)
