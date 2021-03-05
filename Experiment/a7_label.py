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
import tqdm
from joblib import Parallel, delayed


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


def lookup(pattern, df,
           axis=0, case=True, regex=True,
           col_white_list=None, col_black_list=None,
           verbose=False):
    """
    Looks for a given term (string) in a dataframe
    and returns the mask (on the chosen axis) where it was found.

    Parameters:
    -----------
    :pattern: string
        The string (can be a regex) to look for.
    :df: pandas.DataFrame
        The dataframe where the string is looked for.
    :axis: int (0 or 1) or None
        The axis of the desired mask:
            - 0 to get the lines where the term was found
            - 1 to get the columns
            - None to get a 2D mask
    :case: boolean
        If True, the lookup is case sensitive.
    :regex: boolean
        If True, the pattern is matched as a regex.
    :col_white_list: List of strings (or any iterable that returns strings)
        If not None: the names of the columns the lookup will be restricted to.
    :col_black_list: List of strings (or any iterable that returns strings)
        If not None: the names of the columns where the lookup will not occur.
    :verbose: boolean
        If True, information is printed about the lookup.
    """

    # select the proper columns where to look for the term
    if col_white_list:
        df_explored = df[col_white_list]
    else:
        df_explored = df.copy()

    if col_black_list:
        df_explored = df_explored.drop(col_black_list, axis=1)

    df_explored = df_explored.select_dtypes(include=['object'])

    if verbose:
        print("> The term '"
              + pattern
              + "' will be looked for in the following columns:",
              df_explored.columns)

    # does the lookup
    mask = np.column_stack([df_explored[col].astype(str).str.contains(pattern,
                                                          case=case,
                                                          regex=regex,
                                                          na=False)
                            for col in df_explored])
    if verbose:
        print("> Found values:",
              mask.sum())

    if axis is not None:
        if axis == 0:
            mask = mask.any(axis=1)
        else:
            mask = mask.any(axis=0)
        if verbose:
            print("> Found entries along axis", axis,
                  ":", mask.sum())

    return mask


def unit_tagger(pattern, df,
                tag=None, label_col=None, reset=False,
                case=False, regex=False,
                col_white_list=None, col_black_list=None,
                verbose=False):
    """
    Looks for a given term (string) in a dataframe
    and adds a corresponding tag to the rows where it is found.

    Parameters:
    -----------
    :pattern: string
        The string (can be a regex) to look for.
    :df: pandas.DataFrame
        The dataframe where the string is looked for.
    :tag: string or None
        The tag to add if the pattern is found.
        If None, the pattern is used as the tag.
        (try not to use pattern with complex regex as column name)
    :label_col: string or None
        The name of the column where the tag should be added.
        If None, a new column with the name of the tag is created
        and the tag presence is reported as a boolean.
    :reset: boolean
        If True (only relevant if label_col is not None),
        the tag column is set to an empty string.
        (this happens inplace, be careful not to delete useful data)
    :case: boolean
        If True, the lookup is case sensitive.
    :regex: boolean
        If True, the pattern is matched as a regex.
    :col_white_list: List of strings (or any iterable that returns strings)
        If not None: the names of the columns the lookup will be restricted to.
    :col_black_list: List of strings (or any iterable that returns strings)
        If not None: the names of the columns where the lookup will not occur.
    :verbose: boolean
        If True, information is printed about the lookup.
    """
    df_labelled = df
    if tag is None:
        tag = pattern

    if ((label_col is not None)
            and ((label_col not in df.columns)
                 or reset)):
        df_labelled.loc[:, label_col] = ""

    mask = lookup(pattern=pattern, df=df,
                  axis=0, case=case, regex=regex,
                  col_white_list=col_white_list, col_black_list=col_black_list,
                  verbose=verbose)

    if verbose:
        print(">>> Number of tags found for the tag '{}': {}"
              .format(tag, mask.sum()))

    if label_col is not None:
        df_labelled.loc[mask, label_col] = \
            df_labelled.loc[mask, label_col] + tag + ","
    else:
        df_labelled.loc[:, tag] = mask

    return df_labelled


def vocab_tagger(vocab, df,
                 label_col=None,
                 reset=False,
                 case=False,
                 col_white_list=None, col_black_list=None,
                 verbose=False):
    """
    Looks for a given term (string) in a dataframe
    and adds a corresponding tag to the rows where it is found.

    Parameters:
    -----------
    :vocab: list of strings
        The strings to look for.
    :df: pandas.DataFrame
        The dataframe where the string is looked for.
    :label_col: string or None
        The name of the column where the tags should be added.
        If None, a new column with the name of the tag is created for each tag
        and the tag presence is reported as a boolean.
    :reset: boolean
        If True (only relevant if label_col is not None),
        the tag column is set to an empty string.
        (this happens inplace, be careful not to delete useful data)
    :case: boolean
        If True, the lookup is case sensitive.
    :col_white_list: List of strings (or any iterable that returns strings)
        If not None: the names of the columns the lookup will be restricted to.
    :col_black_list: List of strings (or any iterable that returns strings)
        If not None: the names of the columns where the lookup will not occur.
    :verbose: boolean
        If True, information is printed about the lookup.
    """
    df_labelled = df

    for tag in vocab:
        df_labelled = unit_tagger(
            tag, df_labelled,
            tag=None, label_col=label_col, reset=reset,
            case=case, regex=False,
            col_white_list=col_white_list,
            col_black_list=col_black_list,
            verbose=verbose
        )

    return df_labelled


def parallel_vocab_tagger(n_jobs, vocab, df, **kwargs):
    """Parallel version of the `vocab_tagger` function."""
    df_splits = np.array_split(df, n_jobs * 3)

    vocab_tagger_function = partial(vocab_tagger, vocab=vocab, **kwargs)
    df_splits_processed = Parallel(n_jobs)(delayed(vocab_tagger_function)(df=df) for df in df_splits)

    return pd.concat(df_splits_processed, axis=0)


def verif_labels(labels, vocab):
    """
    :param labels: string of comma separated labels
    :param vocab: list of valid labels
    :return: string of comma separated valid labels
    """
    return ", ".join(sorted(intersection(re.split(r"\s?,\s?", labels),
                                         vocab)))


def dumb_tagger(df,
                split_regex=r"[\s-_]+",
                label_col="tags",
                min_chars=3,
                vocab=None,
                keep_figures=False,
                col_white_list=None, col_black_list=None,
                verbose=False):
    """Takes the str columns of a dataframe, splits them
     and considers each separate token as a tag.

    Parameters:
    -----------
    :df: pandas.DataFrame
        The dataframe that will be labelled.
    :split_regex: string
        The regex used to split the strings.
        Ex: r"[\W_]+" (default) to split character strings
                separated by any non-letter/figure character
            r",[\s]*" to split multi-words strings separated by commas
    :label_col: string or None
        The name of the column where the tags should be added.
        If None, a new column is created FOR EACH TAG encountered
        and the tag presence is reported as a boolean.
        (be careful, the None value can result in huge dataframes)
    :min_chars: int > 0
        The minimal number (included) of characters for a tag to be kept.
        (should be at least 1 since you might get empty strings)
    :vocab: list or None
        If not None : the vocabulary the tags should be extracted from
    :keep_figures: boolean
        If True, purely numerical tags are kept, else they are removed.
    :col_white_list: List of strings (or any iterable that returns strings)
        If not None: the names of the columns the lookup will be restricted to.
    :col_black_list: List of strings (or any iterable that returns strings)
        If not None: the names of the columns where the lookup will not occur.
    :verbose: boolean
        If True, information is printed about the lookup.
    """

    # select the proper columns where to look for the term
    if col_white_list:
        df_labelled = df[col_white_list]
    else:
        df_labelled = df.copy()

    if col_black_list:
        df_labelled = df_labelled.drop(col_black_list, axis=1, inplace=True)

    df_labelled = df_labelled.select_dtypes(include=['object'])

    df_res = pd.DataFrame(index=df_labelled.index)

    if verbose:
        print("> The term will be looked for in the folowing columns:",
              df_labelled.columns)

    # Concatenation of all columns into a single one separated by spaces
    full_text = df_labelled.apply(' '.join, axis=1)
    full_text = full_text.str.lower()

    # Splitting with chosen regex
    tags = full_text.str.split(split_regex)

    # Tags cleaning according to criteria
    # remove figures-only tags
    if not keep_figures:
        def figures_only(x):
            return re.fullmatch("[0-9]+", x) is None
        tags = tags.apply(lambda x: list(filter(figures_only, x)))

    # remove too-short tags
    def long_enough(x):
        return len(x) >= min_chars
    tags = tags.apply(lambda x: list(filter(long_enough, x)))

    # trim tags (removes spaces at the beginning/end)
    tags = tags.apply(lambda label_list: [tag.strip()
                                        for tag in label_list])

    # remove tags outside of authorized vocabulary
    if vocab is not None:
        in_vocab = lambda x: x in vocab
        tags = tags.apply(lambda x: list(filter(in_vocab, x)))

    # returns tags either as lists within a single "tag" column
    #                  or as single booleans in per-tag columns
    if label_col:
        df_res[label_col] = tags
    else:
        labels_dummies = pd.get_dummies(
            tags.apply(pd.Series).stack()
        ).sum(level=0)
        df_res = labels_dummies

    return df_res


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
