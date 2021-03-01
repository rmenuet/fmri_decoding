# coding: utf-8

# ===================================================================
# Collect fMRIs and metadata from different sources
# Romuald Menuet - June 2018, updated January 2019
# ===================================================================
# Status: Work in progress / Target: I only, for the moment ;)
#
# Summary: This script collects fMRIs and metadata from Neurovault server
#          as well as from local repositories
# ===================================================================
from pathlib import Path

# 3rd party modules
import argparse
import json
import pandas as pd
from nilearn.datasets import fetch_neurovault


def load_colls(neurovault, verbose=False):
    """
    Loads deduplicated (1 row per collection instead of 1 per fMRI) collections
    metadata into a dataframe

    :param neurovault: nilearn.utils.Bunch
        Bunch variable returned by the fetch_neurovault from nilearn
    :param verbose: bool, default=False
        Activates verbose mode for troubleshooting.

    :return: pandas.DataFrame
        Dataframe with all collections metadata.
    """
    # Build a dataframe from the fetcher's collections metadata
    # (1 row per fMRI)
    colls = pd.DataFrame(neurovault.collections_meta)
    colls = (
        colls.set_index("id")
        .sort_index()
    )

    # Compute the number of downloaded fMRIs per collection
    dl_fmri_per_coll = colls.groupby("id").size()
    dl_fmri_per_coll.name = "downloaded_images"
    colls = colls.join(dl_fmri_per_coll)

    # Remove duplicates (1 row per coll instead of 1 per fMRI)
    colls = (
        colls[~colls.index.duplicated(keep='first')]
        .assign(description=lambda df: df["description"].replace("[\\n|\\r]", " ",
                                                        regex=True))
    )

    if verbose:
        print("  > Unique collections whose fMRIs were succesfully downloaded:",
              colls.shape[0])
        print("  > Collections with the most fMRIs:")
        print(colls[["number_of_images", "downloaded_images", "name"]]
              .sort_values("downloaded_images", ascending=False)
              .head(10))

    return colls


def load_fmris(neurovault):
    """ Loads fMRIs metadata into a dataframe

    :param neurovault: nilearn.utils.Bunch
        Bunch variable returned by the fetch_neurovault from nilearn

    :return: pandas.DataFrame
        Dataframe with all fMRIs metadata
    """
    # Build a dataframe from the fetcher's fMRIs metadata
    fmris = pd.DataFrame(neurovault.images_meta)

    return (
        fmris.set_index("id")
        .sort_index()
        .assign(
            description=lambda df: df["description"].replace("[\\n|\\r]", " ", regex=True)
        )
    )


# ===========================
# === ADDING TAGS FOR HCP ===
# ===========================
def get_hcp_tags(df, task, condition):
    """Gets the tags associated to a given task and condition for HCP

    :param df: pandas.DataFrame
        Dataframe containing the associations.
    :param task: str
        Task to look for.
    :param condition: str
        Condition to look for.

    :return: str
        Tags separated by commas.
    """
    # Relevant tags are the columns names starting from the 3rd
    # (1st is the task 2nd the condition)
    tags = df.columns[2:]

    # Get the array of indexes of the relevant tags
    indexes = (
        df[(df["Task"] == task) & (df["Condition"] == condition)]
        .values[0, 2:]
        .astype(bool)
    )

    # Construct the comma-separated tags string
    return ",".join(tags[indexes])


def add_hcp_tags(fmris, hcp_file):
    # Load HCP labels and convert them to booleans
    hcp_meta = (
        pd.read_csv(hcp_file, sep='\t')
        .replace(1.0, True)
        .fillna(value=False)
    )

    # Get the tags string for each fMRI
    for idx, row in hcp_meta.iterrows():
        tags = get_hcp_tags(hcp_meta, row['Task'], row['Condition'])
        hcp_meta.at[idx, "tags"] = tags

    fmris = fmris.assign(tags_hcp="")
    for idx, row in fmris[fmris["collection_id"] == 4337].iterrows():
        for idx_hcp, row_hcp in hcp_meta.iterrows():
            if (
                (row["task"] == row_hcp["Task"])
                and
                (row["contrast_definition"] == row_hcp["Condition"])
            ):
                fmris.at[idx, "tags_hcp"] = row_hcp["tags"]

    return fmris


# ================================
# === MAIN COLLECTION FUNCTION ===
# ================================
def prepare_collect(global_config=None, verbose=False):
    # --------------
    # --- CONFIG ---
    # --------------

    nv_path = Path(global_config["nv_path"])
    nv_path.mkdir(exist_ok=True)
    meta_path = Path(global_config["meta_path"])
    meta_path.mkdir(exist_ok=True)
    cache_path = Path(global_config["cache_path"])
    cache_path.mkdir(exist_ok=True)

    colls_file = str(meta_path / "colls.csv")
    fmris_file = str(meta_path / "fmris.csv")

    hcp_file = global_config["collect"]["hcp_tags"]
    download_mode = global_config["collect"]["download_mode"]

    if verbose:
        source = "disk" if download_mode == "offline" else "NeuroVault"
        print(f" > Fetch fMRIs from {source}")

    neurovault = fetch_neurovault(
        max_images=None,
        collection_terms={},
        image_terms={},
        data_dir=str(nv_path),
        mode=download_mode,
        verbose=2,
    )

    if verbose:
        print(" > Load NeuroVault metadata and images")

    neurovault_collections = load_colls(neurovault, verbose)
    neurovault_fmris = load_fmris(neurovault)

    # Removal of Neurovault's first and latest IBC version (to match Romuald's original data):
    IBC_DUPLICATES_TO_REMOVE = [2138, 4438] # 6618
    fmris = neurovault_fmris.loc[lambda df: ~df.collection_id.isin(IBC_DUPLICATES_TO_REMOVE)]
    colls = neurovault_collections.drop(IBC_DUPLICATES_TO_REMOVE, axis=0)

    if verbose:
        print(" > Adding tags from HCP")

    fmris = add_hcp_tags(fmris, hcp_file)

    if verbose:
        print(">>> Data collection OK, {} fMRIs from Neurovault, {} collections"
              .format(len(fmris), len(colls)))

    # TODO: Add DiFuMo atlas fetching when Nilearn 0.7.1 is published

    # Final dumps
    colls.to_csv(colls_file, header=True)
    fmris.to_csv(fmris_file, header=True)



# execute only if run as a script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to collect fMRIs from Neurovault and extract"
                    " useful metadata.",
        epilog='''Example: python a1_collect.py -C config.json -v'''
    )
    parser.add_argument("-C", "--configuration",
                        default="./preparation_config.json",
                        help="Path of the JSON configuration file")
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Activates (many) outputs")

    args = parser.parse_args()

    with open(args.configuration, encoding='utf-8') as f:
        global_config = json.load(f)

    prepare_collect(global_config, args.verbose)
