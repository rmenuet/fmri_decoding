# coding: utf-8

# ===================================================================
# Extracts fMRIs' labels
# Romuald Menuet - May 2019
# ===================================================================
# Summary: This script extracts labels from metadata
#          that are then used for decoding
# ===================================================================

# 3rd party modules
import argparse
import json
import pickle
import re
from functools import partial

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.utils.dataframe import (
    dumb_tagger,
    parallel_vocab_tagger,
    lookup,
)


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def remove_matching(array, term):
    """
    Equivalent to `return array[array != term]` for lists
    """
    return [x
            for x in array
            if x != term]


def replace_matching(array, term_orig, term_rep):
    """
    Equivalent to `array[array == term_orig] = term_rep;
                   return array` for lists
    """
    return [x
            if x != term_orig
            else term_rep
            for x in array]


def verif_labels(labels, vocab):
    """
    :param labels: string of comma separated labels
    :param vocab: list of valid labels
    :return: string of comma separated valid labels
    """
    return ", ".join(sorted(intersection(re.split(r"\s?,\s?", labels),
                                         vocab)))


def remove_contrast_term_from_column(df, target_column, contrast_regex):
    """Search potential negative contrast term based on contrast regex and remove it"""
    return df.assign(**{
        target_column: lambda df: (
            df[target_column]
            .astype(str)
            .str.lower()
            .str.split(contrast_regex)
            .apply(lambda x: ("" if type(x) is not list else x[0]))
        )
    })


def remove_duplicate_labels(fmris, vocab, n_jobs):
    def _remove_duplicates(fmris, vocab):
        verif_labels_for_vocab = partial(verif_labels, vocab=vocab)

        return fmris.assign(
            labels_no_heuristic=lambda df: df.labels_no_heuristic.astype(str).apply(verif_labels_for_vocab),
            labels_no_task=lambda df: df.labels_no_task.astype(str).apply(verif_labels_for_vocab),
            labels_all=lambda df: df.labels_all.astype(str).apply(verif_labels_for_vocab),
        )

    df_splits = np.array_split(fmris, n_jobs * 3)

    df_splits_processed = Parallel(n_jobs)(
        delayed(partial(_remove_duplicates, vocab=vocab))(fmris=df) for df in df_splits
    )

    return pd.concat(df_splits_processed, axis=0)


def encompass_labels_from_manual_hierarchy(fmris, config, n_jobs):
    def _encompass_labels(fmris, config):
        for _ in range(4):
            # Run 4 times because of hypernymy graph max depth
            for word, hypernyms in config["hypernyms"].items():
                fmris["labels_no_heuristic"] = (
                    fmris["labels_no_heuristic"]
                        .str
                        .split(r"\s?,\s?")
                        .apply(lambda x:
                               replace_matching(x, word, word + "," + hypernyms))
                        .str
                        .join(",")
                )
                fmris["labels_no_task"] = (
                    fmris["labels_no_task"]
                        .str
                        .split(r"\s?,\s?")
                        .apply(lambda x:
                               replace_matching(x, word, word + "," + hypernyms))
                        .str
                        .join(",")
                )
                fmris["labels_all"] = (
                    fmris["labels_all"]
                        .str
                        .split(r"\s?,\s?")
                        .apply(lambda x:
                               replace_matching(x, word, word + "," + hypernyms))
                        .str
                        .join(",")
                )

        return fmris

    df_splits = np.array_split(fmris, n_jobs * 3)

    df_splits_processed = Parallel(n_jobs)(
        delayed(partial(_encompass_labels, config=config))(fmris=df) for df in df_splits
    )

    return pd.concat(df_splits_processed, axis=0)


def prepare_label(global_config=None, n_jobs=1, verbose=False):
    # --------------
    # --- CONFIG ---
    # --------------
    config          = global_config["label"]
    meta_path       = global_config["meta_path"]
    fmris_meta_file = meta_path + global_config["meta_file"]

    # Add output target.
    # TODO: Fix arguments that should be either in the .conf or the CLI
    output_path = "../output/a7_label.csv"
    mask_labels_file = "../output/fmris_masked_labels_file.p"
    labels_mat = "../output/labels_mat.csv"

    # -------------------------------
    # --- LOAD DATA & INIT FIELDS ---
    # -------------------------------
    # get fMRIs metadata from where to extract labels
    fmris_kept = (
        pd.read_csv(fmris_meta_file, low_memory=False, index_col=0)
        .loc[lambda df: df.kept]
    )

    print("> total number of fMRIs to be labelled =", len(fmris_kept))

    # Initialize various label fields that will then be filled
    fmris_kept = fmris_kept.assign(
        labels_from_tags="",
        labels_from_contrasts="",
        labels_from_all="",
        labels_from_tasks="",
        labels_from_rules="",
    )
    fmris_train = (
        fmris_kept.loc[~fmris_kept["collection_id"].isin(config["id_test"])]
       .copy()
    )
    fmris_test = (
        fmris_kept.loc[fmris_kept["collection_id"].isin(config["id_test"])]
        .copy()
        .assign(tags=lambda df: df.tags.str.replace("_", " "))
    )

    # -------------------------------
    # --- EXTRACT ORIGINAL LABELS ---
    # -------------------------------
    # Extract concepts from the full CognitiveAtlas JSON as a "vocab" array
    # (sorted, lower-case & deduplicated concept names)
    concepts = pd.read_json(config["concepts_file"])
    vocab = np.sort(concepts["name"].str.lower().unique())

    # Removing anything that comes after
    # vs, versus, >, minus, neg_, "neg ", no or non
    if verbose:
        print("  > Removing negative parts in contrasts, names and file names")

    fmris_train = remove_contrast_term_from_column(
        fmris_train,
        target_column="contrast_definition",
        contrast_regex=r">|\s-\s|vs|versus|minus|[\s_-]non?[\s_-]|neg[\s_-]",
    )

    fmris_train = remove_contrast_term_from_column(
        fmris_train,
        target_column="name",
        contrast_regex=r">|\s-\s|vs|versus|minus|[ _-]non?[\s_-]|[ _-]neg[\s_-]",
    )

    fmris_train = remove_contrast_term_from_column(
        fmris_train,
        target_column="file",
        contrast_regex=r"vs_|versus_|minus_|[ _-]neg_|[ _-]non?_",
    )

    # Get any phrase matching a CognitiveAtlas concepts in fMRI tags
    if verbose:
        print("  > adding labels from tags")

    fmris_train = parallel_vocab_tagger(n_jobs, vocab, fmris_train,
                               label_col="labels_from_tags",
                               col_white_list=["tags", ])
    fmris_test = parallel_vocab_tagger(n_jobs, vocab, fmris_test,
                              label_col="labels_from_tags",
                              col_white_list=["tags", ])

    # Get labels from contrasts
    if verbose:
        print("  > labelling from contrasts")

    fmris_train = parallel_vocab_tagger(
        n_jobs,
        vocab,
        fmris_train,
        label_col="labels_from_contrasts",
        col_white_list=["contrast_definition"]
    )

    # Get any phrase matching a CognitiveAtlas concepts
    # in all fMRI metadata fields except contrasts
    # and tags that are parsed separately
    if verbose:
        print("  > adding labels from regexp in all fields "
              "(except contrasts and tags)")

    fmris_train = parallel_vocab_tagger(
        n_jobs,
        vocab,
        fmris_train,
        label_col="labels_from_all",
        col_black_list=["contrast_definition", "tags"]
    )

    # -------------------
    # --- ENRICHMENTS ---
    # -------------------
    # Enrich tags using specific rules
    if config["apply_rules"]:
        if verbose:
            print("  > adding tags from specified rules")

        rules = config["rules"]
        for rule in rules:
            if verbose:
                print("    - rule", rule["name"])

            rule_mask = (fmris_train[str(rule["filter"]["field"])]
                         .astype(str)
                         .str
                         .contains(str(rule["filter"]["pattern"]),
                                   case=False)).values

            not_found_column_flag = False
            if rule["rule"]["field"] == "ALL":
                label_mask = lookup(str(rule["rule"]["pattern"]),
                                    fmris_train,
                                    case=False)
            else:
                if str(rule["rule"]["field"]) in fmris_train.columns:
                    label_mask = (fmris_train[str(rule["rule"]["field"])]
                                  .astype(str)
                                  .str
                                  .contains(str(rule["rule"]["pattern"]),
                                            case=False)).values
                else:
                    not_found_column_flag = True
                    label_mask = np.array([False] * len(fmris_train)).astype(bool)
            if verbose:
                print("    |-> matched ",
                      (rule_mask & label_mask).sum(),
                      "times with flag ", not_found_column_flag)

            fmris_train.loc[rule_mask & label_mask, "labels_from_rules"] = (
                    fmris_train.loc[rule_mask & label_mask,
                                    "labels_from_rules"].astype(str)
                    + ","
                    + str(rule["rule"]["labels"])
                    + ","
            )

    # Infer concepts from tasks using the previously loade dictionary
    if verbose:
        print("  > adding labels from tasks")

    # Load a CSV with CognitiveAtlas Task->Conceptsmapping as a dictionary
    # {"task": "concept1, concept2..."}
    if config["apply_task_concept_map"]:
        if verbose:
            print("  > mapping concepts to known tasks")
        task_concept = pd.read_csv(config["task_concept_map_file"],
                                   low_memory=False, index_col=0)
        task_concept["tags"] = (task_concept["Concept 1"] + ","
                                + task_concept["Concept 2"] + ","
                                + task_concept["Concept 3"] + ",")
        task2concept = task_concept["tags"].to_dict()

        fmris_train["labels_from_tasks"] = \
            fmris_train["cognitive_paradigm_cogatlas"].map(task2concept)
        fmris_test["labels_from_tasks"] = \
            fmris_test["cognitive_paradigm_cogatlas"].map(task2concept)

        # Replace np.nan values with empty strings
        fmris_train.loc[fmris_train["labels_from_tasks"].isna(),
                        "labels_from_tasks"] = ""
        fmris_test.loc[fmris_test["labels_from_tasks"].isna(),
                       "labels_from_tasks"] = ""

    # Append tags from Neurovault and IBC fMRIs
    fmris_kept = fmris_train.append(fmris_test)
    fmris_kept.sort_index(inplace=True)

    # Concatenate the concepts
    fmris_kept["labels_no_heuristic"] = (
            fmris_kept["labels_from_tags"] +
            fmris_kept["labels_from_contrasts"] +
            fmris_kept["labels_from_all"]
    )
    fmris_kept["labels_no_task"] = (
            fmris_kept["labels_no_heuristic"] +
            fmris_kept["labels_from_rules"]
    )
    fmris_kept["labels_all"] = (
            fmris_kept["labels_no_task"] +
            fmris_kept["labels_from_tasks"]
    )

    # Verify that all concepts are in the vocabulary and deduplicate them
    if verbose:
        print("  > Removing duplicate labels (before syn/hyp completion)")

    fmris_kept = remove_duplicate_labels(fmris_kept, vocab, n_jobs=n_jobs)

    if config["apply_corrections"]:
        if verbose:
            print("  > Fixing some mistakes (esp. in HCP)")

        corrections = config["corrections"]
        for corr in corrections:
            corr_mask = (fmris_kept[str(corr["filter"]["field"])]
                         .astype(str)
                         .str
                         .contains(str(corr["filter"]["pattern"]),
                                   case=False)).values

            if corr["rule"]["field"] == "ALL":
                label_mask = lookup(str(corr["rule"]["pattern"]),
                                    fmris_kept,
                                    case=False)
            else:
                label_mask = (fmris_kept[str(corr["rule"]["field"])]
                              .astype(str)
                              .str
                              .contains(str(corr["rule"]["pattern"]),
                                        case=False)).values

            if verbose:
                print("    |-> matched",
                      (corr_mask & label_mask).sum(),
                      "times")

            fmris_kept.loc[corr_mask & label_mask, "labels_all"] = (
                    fmris_kept.loc[corr_mask & label_mask,
                                   "labels_all"]
                    .astype(str)
                    .str
                    .replace(corr["rule"]["remove"], "")
            )

    fmris_kept["labels_no_heuristic"].to_csv(f"{output_path[:-4]}_no_heuristic.csv", header=True)
    fmris_kept["labels_no_task"].to_csv(f"{output_path[:-4]}_no_task.csv", header=True)
    fmris_kept["labels_all"].to_csv(f"{output_path[:-4]}_all.csv", header=True)

    # Remove "irrelevant" labels
    # ('concept', 'rule'...)
    if config["apply_remove"]:
        if verbose:
            print("  > Removing some fuzzy labels")

        for to_remove in config["to_remove"]:
            fmris_kept["labels_no_heuristic"] = (
                fmris_kept["labels_no_heuristic"]
                .str
                .split(r"\s?,\s?")
                .apply(lambda x: remove_matching(x, to_remove))
                .str
                .join(",")
            )
            fmris_kept["labels_no_task"] = (
                fmris_kept["labels_no_task"]
                .str
                .split(r"\s?,\s?")
                .apply(lambda x: remove_matching(x, to_remove))
                .str
                .join(",")
            )
            fmris_kept["labels_all"] = (
                fmris_kept["labels_all"]
                .str
                .split(r"\s?,\s?")
                .apply(lambda x: remove_matching(x, to_remove))
                .str
                .join(",")
            )

    # Replace synonyms
    # ('audition' and 'auditory perception'...)
    if config["apply_synonyms"]:
        if verbose:
            print("  > Merging some synonyms")

        for source, target in config["synonyms"].items():
            fmris_kept["labels_no_heuristic"] = (
                fmris_kept["labels_no_heuristic"]
                .str
                .split(r"\s?,\s?")
                .apply(lambda x: replace_matching(x, source, target))
                .str
                .join(",")
            )
            fmris_kept["labels_no_task"] = (
                fmris_kept["labels_no_task"]
                .str
                .split(r"\s?,\s?")
                .apply(lambda x: replace_matching(x, source, target))
                .str
                .join(",")
            )
            fmris_kept["labels_all"] = (
                fmris_kept["labels_all"]
                .str
                .split(r"\s?,\s?")
                .apply(lambda x: replace_matching(x, source, target))
                .str
                .join(",")
            )

    # Infer encompassing labels based on hyperonyms
    # ('reading': add 'semantic processing' and 'visual perception'...)
    if config["apply_hypernyms"]:
        if verbose:
            print("  > Infering encompassing labels based on manual hierarchy")

        for _ in range(4):
            # Run 4 times because of hypernymy graph max depth
            for word, hypernyms in config["hypernyms"].items():
                fmris_kept["labels_no_heuristic"] = (
                    fmris_kept["labels_no_heuristic"]
                    .str
                    .split(r"\s?,\s?")
                    .apply(lambda x:
                        replace_matching(x, word, word + "," + hypernyms))
                    .str
                    .join(",")
                )
                fmris_kept["labels_no_task"] = (
                    fmris_kept["labels_no_task"]
                    .str
                    .split(r"\s?,\s?")
                    .apply(lambda x:
                        replace_matching(x, word, word + "," + hypernyms))
                    .str
                    .join(",")
                )
                fmris_kept["labels_all"] = (
                    fmris_kept["labels_all"]
                    .str
                    .split(r"\s?,\s?")
                    .apply(lambda x:
                        replace_matching(x, word, word + "," + hypernyms))
                    .str
                    .join(",")
                )


    # Verify that all concepts are in the vocabulary and deduplicate them
    if verbose:
        print("  > Removing duplicate labels (after syn/hyp completion)")
    fmris_kept["labels_no_heuristic"] = (
        fmris_kept["labels_no_heuristic"]
        .astype(str)
        .apply(lambda x: verif_labels(x, vocab))
    )
    fmris_kept["labels_no_task"] = (
        fmris_kept["labels_no_task"]
        .astype(str)
        .apply(lambda x: verif_labels(x, vocab))
    )
    fmris_kept["labels_all"] = (
        fmris_kept["labels_all"]
        .astype(str)
        .apply(lambda x: verif_labels(x, vocab))
    )
    # Save result as CSV
    fmris_kept["labels_no_heuristic"].to_csv(f"{output_path[:-4]}_syn_hyp.csv", header=True)
    fmris_kept["labels_no_task"].to_csv(f"{output_path[:-4]}_no_task_syn_hyp.csv", header=True)
    fmris_kept["labels_all"].to_csv(f"{output_path[:-4]}_all_syn_hyp.csv", header=True)
    fmris_kept["labels_from_tags"].to_csv(f"{output_path[:-4]}_from_tags.csv", header=True)
    fmris_kept["labels_from_contrasts"].to_csv(f"{output_path[:-4]}_from_contrasts.csv", header=True)
    fmris_kept["labels_from_all"].to_csv(f"{output_path[:-4]}_from_all.csv", header=True)
    fmris_kept["labels_from_tasks"].to_csv(f"{output_path[:-4]}_from_tasks.csv", header=True)
    fmris_kept["labels_from_rules"].to_csv(f"{output_path[:-4]}_from_rules.csv", header=True)

    # Save labels mask
    mask_labels_no_heuristic = ~fmris_kept["labels_no_heuristic"].isna().values
    with open(mask_labels_file[:-2] + "no_heuristic.p", 'wb') as f:
        pickle.dump(mask_labels_no_heuristic, f)
    mask_labels = (~fmris_kept["labels_all"].isna()).values
    with open(mask_labels_file, 'wb') as f:
        pickle.dump(mask_labels, f)

    # Build and save labels matrix
    labels_present_no_heuristic = fmris_kept[mask_labels_no_heuristic][["labels_no_heuristic"]]
    Y_labels_no_heuristic = dumb_tagger(
        labels_present_no_heuristic,
        split_regex=r"\s?,\s?",
        vocab=vocab,
        label_col=None,
    )
    Y_labels_no_heuristic.to_csv(f"{labels_mat[:-4]}no_heuristic.csv", header=True)
    labels_present = fmris_kept[mask_labels][["labels_all"]]
    Y_labels = dumb_tagger(
        labels_present,
        split_regex=r"\s?,\s?",
        vocab=vocab,
        label_col=None,
    )
    Y_labels.to_csv(labels_mat, header=True)

    if verbose:
        print("> total number of labelled fMRIs =",
              (fmris_kept["labels_all"].str.len() > 0).sum())


# execute only if run as a script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to add tags to fMRIs fetched from Neurovault.",
        epilog='''Example: python a7_label.py -C config.json -v'''
    )
    parser.add_argument("-C", "--config",
                        default="./config.json",
                        help="Path of the JSON config file")
    parser.add_argument("-j", "--jobs",
                        type=int,
                        default=1,
                        help="Number of jobs")
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Activates (many) debugging outputs")

    args = parser.parse_args()

    with open(args.config) as f:
        global_config = json.load(f)

    prepare_label(global_config, args.jobs, args.verbose)
