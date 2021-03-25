from functools import partial

import numpy as np
import pandas as pd
import re
from joblib import delayed, Parallel


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
