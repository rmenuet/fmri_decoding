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
import pickle
import pandas as pd
from nilearn.datasets import fetch_neurovault


# ================================
# === FETCHING FROM NEUROVAULT ===
# ================================
def fetch_nv(repo, nv_file,
             download=False,
             verbose=False,
             mode="download_new"):
    """
    Loads neurovault into memory, either downloading it from the web-API or
    loading it from the disk.

    :param repo: str
        Path where the data is downloaded.
    :param nv_file: str
        Pickle file where the full data is saved
        (for faster loading than the fetch_neurovault).
    :param download: bool, default=False
        If True: the data is downloaded from the web-API.
    :param verbose: bool, default=False
        Activate verbose mode.

    :return: Bunch
        A dict-like object containing the data from fMRIs fetched from
        Neurovault.
    """

    # Download and save to disk or load from disk
    if download:
        if verbose:
            print("...Download from Neurovault API...")

        neurovault = fetch_neurovault(max_images=None,
                                      collection_terms={},
                                      image_terms={},
                                      data_dir=repo,
                                      mode=mode,
                                      verbose=2)
        with open(nv_file, 'wb+') as f:
            pickle.dump(neurovault, f)
    else:
        if verbose:
            print("...Load pre-fetched data from Neurovault...")
        with open(nv_file, 'rb') as f:
            neurovault = pickle.load(f)

    n_fmri_dl = len(neurovault.images)
    if verbose:
        print("  > Number of (down)loaded fmri =", n_fmri_dl)

    return neurovault


def load_colls(nv_file,
               colls_file,
               verbose=False):
    """
    Loads deduplicated (1 row per collection instead of 1 per fMRI) collections
    metadata into a dataframe

    :param nv_file: str
        Pickle file where the full data is saved
        (for faster loading than the fetch_neurovault).
    :param colls_file: str
        CSV file  where to save collections metadata.
    :param verbose: bool, default=False
        Activates verbose mode for troubleshooting.

    :return: pandas.DataFrame
        Dataframe with all collections metadata.
    """

    with open(nv_file, 'rb') as f:
        neurovault = pickle.load(f)

    # Build a dataframe from the fetcher's collections metadata
    # (1 row per fMRI)
    colls = pd.DataFrame(neurovault.collections_meta)
    colls.set_index("id", inplace=True)
    colls.sort_index(inplace=True)

    # Compute the number of downloaded fMRIs per collection
    dl_fmri_per_coll = colls.groupby("id").size()
    dl_fmri_per_coll.name = "downloaded_images"
    colls = colls.join(dl_fmri_per_coll)

    # Remove duplicates (1 row per coll instead of 1 per fMRI)
    colls = colls[~colls.index.duplicated(keep='first')]

    # Remove linebreaks in descriptions to better handle them as CSV afterwards
    colls["description"] = colls["description"].replace("[\\n|\\r]", " ",
                                                        regex=True)

    if verbose:
        print("  > Unique collections whose fMRIs were succesfully downloaded:",
              colls.shape[0])
        print("  > Collections with the most fMRIs:")
        print(colls[["number_of_images", "downloaded_images", "name"]]
              .sort_values("downloaded_images", ascending=False)
              .head(10),
              "\n")

    colls.to_csv(colls_file, header=True)

    return colls


def load_fmris(nv_file,
               fmris_file):
    """ Loads fMRIs metadata into a dataframe

    :param nv_file: str
        Pickle file where the full data is saved
        (for faster loading than the fetch_neurovault)
    :param fmris_file: str
        CSV file  where to save fMRIs metadata

    :return: pandas.DataFrame
        Dataframe with all fMRIs metadata
    """

    with open(nv_file, 'rb') as f:
        neurovault = pickle.load(f)

    # Build a dataframe from the fetcher's fMRIs metadata
    fmris = pd.DataFrame(neurovault.images_meta)
    fmris.set_index("id", inplace=True)
    fmris.sort_index(inplace=True)

    # Remove linebreaks in descriptions to better handle them as CSV afterwards
    fmris["description"] = fmris["description"].replace("[\\n|\\r]", " ",
                                                        regex=True)

    fmris.to_csv(fmris_file, header=True)

    return fmris


# ===========================
# === ADDING TAGS FOR HCP ===
# ===========================
def get_hcp_tags(df, task, condition):
    """ Gets the tags associated to a given task and condition for HCP

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
    indexes = df[(df["Task"] == task)
                 &
                 (df["Condition"] == condition)].values[0, 2:]
    indexes = indexes.astype(bool)

    # Construct the comma-separated tags string
    labels_str = ""
    for tag in tags[indexes]:
        labels_str = labels_str + tag + ","

    return labels_str[:-1]


def add_hcp_tags(fmris_file, hcp_file):
    # Load fMRIs metadata (as loaded from Neurovault)
    fmris = pd.read_csv(fmris_file, low_memory=False, index_col=0)

    # Load HCP labels and convert them to booleans
    hcp_meta = pd.read_csv(hcp_file, sep='\t')
    hcp_meta.replace(1.0, True, inplace=True)
    hcp_meta.fillna(value=False, inplace=True)

    # Get the tags string for each fMRI
    for idx, row in hcp_meta.iterrows():
        tags = get_hcp_tags(hcp_meta, row['Task'], row['Condition'])
        hcp_meta.at[idx, "tags"] = tags

    fmris["tags_hcp"] = ""
    for idx, row in fmris[fmris["collection_id"] == 4337].iterrows():
        for idx_hcp, row_hcp in hcp_meta.iterrows():
            if (
                (row["task"] == row_hcp["Task"])
                and
                (row["contrast_definition"] == row_hcp["Condition"])
            ):
                fmris.at[idx, "tags_hcp"] = row_hcp["tags"]

    fmris.to_csv(fmris_file, header=True)

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

    nv_file = str(cache_path / "nv_meta.p")
    colls_file = str(meta_path / "colls.csv")
    fmris_file = str(meta_path / "fmris.csv")

    hcp_file = global_config["collect"]["hcp_tags"]
    download = global_config["collect"]["download"]
    download_mode = global_config["collect"]["download_mode"]

    # -----------------------------
    # --- FETCH FROM NEUROVAULT ---
    # -----------------------------
    if verbose:
        print("=" * 30)
        source = "website" if download else "disk"
        print(f" > Fetching Neurovault data from the {source}")

    fetch_nv(str(nv_path), nv_file, download, verbose, mode=download_mode)

    colls_nv = load_colls(nv_file, colls_file, verbose)
    fmris_nv = load_fmris(nv_file, fmris_file)

    # Removal of Neurovault's first IBC version (reuploaded in better quality):
    # TODO: migrer dans conf exp
    FIRST_IBC_COLLECTION_TO_REMOVE = 2138
    fmris = (
        pd.read_csv(fmris_file, low_memory=False, index_col=0)
        .loc[lambda df: df.collection_id != FIRST_IBC_COLLECTION_TO_REMOVE]
    )
    fmris.to_csv(fmris_file, header=True)
    colls = (
        pd.read_csv(colls_file, low_memory=False, index_col=0)
        .drop(FIRST_IBC_COLLECTION_TO_REMOVE, axis=0)
    )
    colls.to_csv(colls_file, header=True)

    # ------------------------------------------
    # --- ADDING HCP TAGS FROM SEPARATE BASE ---
    # ------------------------------------------
    if verbose:
        print(" > Adding tags from HCP")

    add_hcp_tags(fmris_file, hcp_file)

    if verbose:
        print(">>> Data collection OK, {} fMRIs from Neurovault, {} collections"
              .format(len(fmris_nv), len(colls_nv)))


# execute only if run as a script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to collect fMRIs from Neurovault and extract"
                    " useful metadata.",
        epilog='''Example: python a1_collect.py -C config.json -v'''
    )
    parser.add_argument("-C", "--configuration",
                        default="./config.json",
                        help="Path of the JSON configuration file")
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Activates (many) outputs")

    args = parser.parse_args()

    with open(args.configuration, encoding='utf-8') as f:
        global_config = json.load(f)

    prepare_collect(global_config, args.verbose)
