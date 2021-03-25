# coding: utf-8

# ===================================================================
# Decoding experiment for preprocessed fMRIs
# Romuald Menuet - November 2018
# ===================================================================
# Status: Work in progress / Target: I only, for the moment ;)
#
# Summary: This script runs a decoding experiment on preprocessed fMRIs
#          todo: virer useless functions
# ===================================================================

#%% IMPORTS
# sys & IO
import argparse
import datetime
import json
import os
import pickle
import shutil
from functools import partial
from loguru import logger

# data management
import numpy as np
import pandas as pd
from sklearn import preprocessing

# ML
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC

# custom modules
from src.learning import models
from src.learning.estimators import PytorchEstimator
from src.learning.losses import multinomial_cross_entropy
from src.learning.metrics import mean_auc, recall_n
from src.utils.dataframe import dumb_tagger
from src.utils.file import get_json_files_in_dir, mkdir
from src.tools import (
    mask_rows,
    yes_or_no,
    gridsearch_complexity,
    highly_corr_cols_np,
    one_compact_line,
)


def decoding_experiment(configuration="spec_template.json",
                        mode="train",
                        folder="",
                        results_file="",
                        used_gpu=0,
                        n_jobs=1,
                        verbose=False,
                        plot_training=False,
                        force=False):
    """fMRI decoding experiment: transforms metadata (infers labels),
    trains a decoding model, then evaluates and saves it
    following the provided configuration

    :param configuration: str
                          path to experiment JSON configuration file
    :param mode: str ('train' or 'retrain' or 'evaluate')
                 execution mode:
                  - 'train': transform data, train & evaluate model
                  - 'retrain': (re)train & evaluate model (to be implemented)
                  - 'train': load & evaluate model (to be implemented)
    :param folder: str
                   root of the folder where to store data
    :param results_file: str
                         name of the file where to store results
    :param used_gpu: int (>= -1)
                     ID of the CUDA device to be used
                     (if -1 model is trained on CPU)
    :param n_jobs: int (>= 1)
                   number of parallel jobs for grid search
    :param verbose: bool
                    print debugging information & results
    :param plot_training: bool
                          plot learning curve
    :param force: bool
                  don't ask if previous model & results will be overwritten

    :return: nothing for the moment
    """

    # ID of the experiment used in reports
    #   (filename without JSON extension)
    ID = configuration[:-5].split("/")[-1]

    logger.info(f"=== Decoding experiment {ID} ===")

    with open(configuration, encoding='utf-8') as f:
        config = json.load(f)

    # Path where to save transformed data, model and results
    path = folder + ID + "/"
    per_label_results_file = path + "per_label_results.csv"

    # Control whether the experiment was already run
    # and results mught be overwritten
    should_ask_question_for_experiment = (
        (not force)
        &
        ((not mkdir(path)) & (mode == "train"))
    )
    if should_ask_question_for_experiment:
        if not yes_or_no("\nExperiment already run, "
                         "ARE YOU SURE you want to retrain model? "):
            print("Execution aborted")
            return 0

    # Copy original configuration to backup folder
    shutil.copy(configuration, path + configuration.split("/")[-1])

    # If GPU is used, limit CUDA visibility to this device
    if used_gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(used_gpu)
        used_gpu = 0
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        used_gpu = -1

    # -------------------------
    # --- MODELS DEFINITION ---
    # -------------------------
    model_classes = {
        "sk_logreg": LogisticRegression,
        "sk_logreg_cv": LogisticRegressionCV,
        "sk_svm": SVC,
        "sk_lda": LinearDiscriminantAnalysis,
    }

    # -------------------------
    # --- LOSSES DEFINITION ---
    # -------------------------
    loss_functions = {
        "logreg": torch.nn.BCEWithLogitsLoss(reduction='none'),
        "multinomial": multinomial_cross_entropy,
    }

    # ---------------------------
    # --- TRAINING PARAMETERS ---
    # ---------------------------
    # Grid to explore
    param_grid = config["grid_params"]
    exploratory_comp = gridsearch_complexity(param_grid)

    logger.info(f"Gridsearch size: {exploratory_comp}")

    # Initial values for model instanciation
    param_ini = {k: v[0] for (k, v) in param_grid.items()}

    # --------------------
    # --- DATA LOADING ---
    # --------------------
    # Metadata loading
    meta = (
        pd.read_csv(config["data"].get("meta_file"), low_memory=False, index_col=0)
        .loc[lambda df: df.kept]
    )

    # Labels vocabulary loading
    with open(config["data"]["concepts_file"], encoding='utf-8') as f:
        concept_names = [line.rstrip('\n') for line in f]
        concept_names = sorted([concept_name.strip().lower()
                         for concept_name in concept_names])

    # Features loading
    with open(config["data"].get("features_file"), 'rb') as f:
        X = pickle.load(f)

    # Samples' labels loading
    labels = pd.read_csv(config["data"]["labels_file"], low_memory=False, index_col=0)

    logger.info("Data loaded")
    logger.info(f"Number of kept fMRIs in dataset {len(meta)}")

    # Only keep samples (fMRIs metadata and their embeddings) with labels
    mask_labelled = ~labels.iloc[:, 0].isna()
    meta, X, labels = mask_rows(mask_labelled, meta, X, labels)

    # Target as a one-hot encoding of labels
    Y = dumb_tagger(labels,
                    split_regex=r",\s*",
                    vocab=concept_names,
                    label_col=None)

    # Extract vocabulary of labels present in the dataset
    vocab_orig = np.array(Y.columns)
    logger.info(f"Number of labels in the whole dataset (TRAIN+TEST): {len(vocab_orig)}")

    # Convert Y to np.array of int
    Y = Y.values * 1

    # In case the labels did not come from the proper vocabulary,
    #   remove the fmris without any label
    mask_label_checked = (Y.sum(axis=1) != 0)
    meta, X, Y = mask_rows(mask_label_checked, meta, X, Y)
    
    # Remove maps from blacklist if present
    if config["data"].get("blacklist"):
        mask_not_blacklisted = np.full(len(meta), True)
        blacklist = config["data"].get("blacklist")
        for blacklist_key in blacklist:
            mask_not_blacklisted = (
                mask_not_blacklisted
                &
                ~meta[blacklist_key].isin(blacklist[blacklist_key])
            )
        meta, X, Y = mask_rows(mask_not_blacklisted, meta, X, Y)

    logger.info(f"Number of fMRIs with labels: {len(meta)}")
    logger.info(f"Number of labels in Train: {len(vocab_orig)}")

    # Filtering labels with too few instances in train
    mask_test = (meta["collection_id"].isin(config["evaluation"]["test_IDs"]))
    colmask_lab_in_train = (Y[~mask_test].sum(axis=0)
                            >= config["labels"]["min_train"])

    number_of_rare_labels = len(vocab_orig) - int(colmask_lab_in_train.sum())
    logger.info(f"Removed {number_of_rare_labels} labels that were too rare")

    # updating X and Y
    Y = Y[:, colmask_lab_in_train]
    mask_lab_in_train = (np.sum(Y, axis=1) != 0)
    meta, X, Y = mask_rows(mask_lab_in_train, meta, X, Y)

    # updating vocab mask
    vocab_current = vocab_orig[colmask_lab_in_train]

    # Remove almost fully correlated columns
    labels_low_corr_indices = highly_corr_cols_np(Y,
                                                  vocab_current,
                                                  0.95,
                                                  True)

    number_of_too_correlated_labels = Y.shape[1] - len(labels_low_corr_indices)
    logger.info(f"Removed {number_of_too_correlated_labels} labels that were too correlated")

    Y = Y[:, labels_low_corr_indices]
    vocab_current = vocab_current[labels_low_corr_indices]

    # Update of data and testset mask after highly correlated labels removal
    mask_has_low_corr_lab = (np.sum(Y, axis=1) != 0)
    meta, X, Y = mask_rows(mask_has_low_corr_lab, meta, X, Y)
    mask_test = meta["collection_id"].isin(config["evaluation"]["test_IDs"])
    
    # save original version of labels to predict before labels inference
    Y_orig = Y.copy()
    
    logger.info(f"Number of kept labels: {Y.shape[1]}")

    # Concept values transformations
    if (config["labels"].get("transformation") == "none"
            or config["labels"].get("transformation") is None):
        pass
    elif config["labels"].get("transformation") == "thresholding":
        Y = (Y >= config["labels"]["threshold"]) * 1
    elif config["labels"].get("transformation") == "normalization":
        Y = Y / Y.sum(axis=1, keepdims=True)
    else:
        raise ValueError("Unsupported transformation of concept values")

    # ------------------------------
    # --- FEATURES PREPROCESSING ---
    # ------------------------------
    # Thresholding
    if config["data"].get("positive_values"):
        X[X < 0] = 0

    # Train/valid split
    X_train, Y_train_orig, Y_train = mask_rows(~mask_test, X, Y_orig, Y)
    X_test, Y_test_orig, Y_test = mask_rows(mask_test, X, Y_orig, Y)
    indices_train = list(meta[~mask_test].index)
    indices_test = list(meta[mask_test].index)

    # Scaling over features or samples based on train dataset
    if config["data"].get("scaling") == "features":
        scaler = preprocessing.StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    elif config["data"].get("scaling") == "samples":
        preprocessing_on_samples = partial(preprocessing.scale, with_mean=True, with_std=True, axis=1)
        X_train = preprocessing_on_samples(X_train)
        X_test = preprocessing_on_samples(X_test)
    elif config["data"].get("scaling") == "max":
        X_train_max = X_train.max()
        X_train = X_train / X_train_max
        X_test = X_test / X_train_max

    # ------------------------
    # --- SAMPLING WEIGHTS ---
    # ------------------------
    # Groups definition (for CV splits, loss reweighting and sampling)
    meta = (
        meta.assign(
            group=lambda df: (df["collection_id"].astype(str) + " " + df["cognitive_paradigm_cogatlas"].astype(str))
        )
    )

    # Reweighting samples if required
    weighted_samples = (
        config["torch_params"].get("group_power")
        and config["torch_params"]["group_power"] < 1.0
    )
    if weighted_samples:
        group_size = meta.groupby("group")["kept"].count()
        group_size.name = "group_size"

        # sampling weights
        group_weights = meta.join(group_size, on="group")["group_size"]
        group_weights = (group_weights ** config["torch_params"]["group_power"] / group_weights)
        sample_weights = group_weights.values.reshape((-1, 1))
        sample_weights_train = sample_weights[~mask_test]
        X_train = np.hstack((sample_weights_train, X_train))

    logger.info("Data preprocessed")
    logger.info(f"Samples kept in TRAIN: {len(X_train)}")
    logger.info(f"Samples kept in TEST: {len(X_test)}")
    if weighted_samples:
        logger.info(f"Min/Max sampling weight in TRAIN: {sample_weights_train.min()} / {sample_weights_train.max()}")

    # -------------------
    # --- SAVING DATA ---
    # -------------------
    for variable_name in ["X_train", "Y_train", "Y_train_orig", "X_test", "Y_test", "Y_test_orig", "indices_train", "indices_test"]:
        output_path_for_variable = f"{path}{variable_name}.p"
        with open(output_path_for_variable, "wb") as f:
            # eval(name) retrieves the variable value which has this name
            pickle.dump(eval(variable_name), f)

    pd.DataFrame(vocab_orig).to_csv(f"{path}vocab_orig.csv")
    pd.DataFrame(vocab_current).to_csv(f"{path}vocab.csv")

    logger.info("Preprocessed data saved")

    # ---------------------------
    # --- MODEL INSTANCIATION ---
    # ---------------------------
    if config.get("estimator_type") == "sklearn":
        estimator_type = "sklearn"
        clf_template = model_classes[config["model_name"]]
    else:
        estimator_type = "pytorch"
        clf_template = PytorchEstimator(
            gpu=(used_gpu > -1),
            used_gpu=used_gpu,
            model_class=getattr(models, config["model_name"]),
            loss_func=loss_functions[config["loss"]["loss_func_name"]],
            epochs=config["torch_params"]["epochs"],
            batch_size=config["torch_params"]["batch_size"],
            adam=config["torch_params"]["Adam"],
            verbose=plot_training*2,
            **param_ini
        )

    # ----------------
    # --- TRAINING ---
    # ----------------
    logger.info("Launch training")

    clf, clf_grid = None, None
    if exploratory_comp == 1:
        logger.info("Single model training...")
        if estimator_type == "sklearn":
            clfs = {}
            for i, concept in enumerate(vocab_current):
                logger.info(f"Training for {concept}")
                clf = clf_template(**param_ini)
                clf.fit(X_train, Y_train[:, i])
                clfs[concept] = clf
        else:
            clf = clf_template
            clf.fit(X_train, Y_train,
                    weighted_samples=weighted_samples,
                    n_jobs=n_jobs)

    else:
        assert estimator_type == "pytorch", "grid search not implemented yet" \
                                            " for sklearn estimators"
        if (
                not config["torch_params"].get("search")
                or
                config["torch_params"]["search"] == -1
                or
                config["torch_params"]["search"] > exploratory_comp
        ):
            if verbose:
                print("  > Cross validation with",
                      config["torch_params"]["splits"],
                      "splits,",
                      exploratory_comp * config["torch_params"]["splits"] + 1,
                      "models fitted")
            clf_grid = GridSearchCV(clf_template, param_grid,
                                    cv=config["torch_params"]["splits"],
                                    iid=False, n_jobs=n_jobs,
                                    verbose=verbose*1)
        else:
            if verbose:
                print(
                    "  > Cross validation with",
                    config["torch_params"]["splits"],
                    "splits,",
                    config["torch_params"]["search"]
                    * config["torch_params"]["splits"] + 1,
                    "models fitted"
                )
            clf_grid = RandomizedSearchCV(
                clf_template, param_grid,
                cv=config["torch_params"]["splits"],
                n_iter=config["torch_params"]["search"],
                iid=False, n_jobs=n_jobs,
                verbose=verbose * 1
            )
        clf_grid.fit(X_train, Y_train,
                     meta.loc[~mask_test]["group"].values,
                     weighted_samples=weighted_samples)

        # Grid search detailed results backup
        pd.DataFrame(clf_grid.cv_results_).to_csv(f"{path}per_param_results.csv")

        clf = clf_grid.best_estimator_

        print("  > Best params:", str(clf_grid.best_params_))
        print("  > Best score:", str(clf_grid.best_score_))
        print("=" * 60)

    # Model backup
    if estimator_type == "sklearn":
        with open(path + "clf.p", 'wb') as f:
            pickle.dump(clfs, f)
    else:
        with open(path + "clf.p", 'wb') as f:
            pickle.dump(clf, f)
        torch.save(clf.model, path + "clf.pt")

    # ------------------
    # --- EVALUATION ---
    # ------------------
    # Loading the Recall threshold N
    # if no N is set, we check that true labels are ranked among the first 10%
    N = config["evaluation"].get("recall@N")
    if N is None:
        N = Y_train.shape[1] // 10

    # Dumb predictions based on labels ranking in training set
    labels_rank_train = np.sum(Y_train, axis=0)
    Y_train_pred_dumb = np.tile(labels_rank_train, [len(Y_train), 1])
    Y_test_pred_dumb = np.tile(labels_rank_train, [len(Y_test), 1])

    # Remove weights from features for final prediction (1st col)
    if weighted_samples:
        X_train = X_train[:, 1:]

    # Predict labels on TEST set with best classifier
    if estimator_type == "sklearn":
        Y_train_pred = np.zeros((len(X_train), len(vocab_current)))
        Y_test_pred = np.zeros((len(X_test), len(vocab_current)))
        for i, concept in enumerate(vocab_current):
            Y_train_pred[:, i] = clfs[concept].predict(X_train)
            Y_test_pred[:, i] = clfs[concept].predict(X_test)

    else:
        Y_train_pred = clf.predict(X_train)
        Y_test_pred = clf.predict(X_test)

    # Compute recalls for dumb and trained classifiers
    weighted_recall_n = partial(recall_n, n=N, reduce_mean=True)
    recall_train_dumb = weighted_recall_n(Y_train_pred_dumb, Y_train_orig)
    recall_test_dumb = weighted_recall_n(Y_test_pred_dumb, Y_test_orig)
    recall_train = weighted_recall_n(Y_train_pred, Y_train_orig)
    recall_test = weighted_recall_n(Y_test_pred, Y_test_orig)

    # Compute AUC for trained classifier
    auc = mean_auc(Y_test_pred, Y_test_orig)

    # Print and save performances
    perf = f"""
    PERFORMANCES:
        DUMB BASELINE:
            Recall@{str(N)} TRAIN: {str(recall_train_dumb)}
            Recall@{str(N)} TEST: {str(recall_test_dumb)}
        MODEL:
            Recall@{str(N)} TRAIN: {str(recall_train)}
            Recall@{str(N)} TEST: {str(recall_test)}
            Mean ROC AUC TEST: {str(auc)}
    """
    print(perf)

    # Print and save recap on experiment
    desc = "Experiment: " + ID
    desc += "\n  Trained on: " + str(datetime.datetime.now())
    desc += "\n  Scaling: " + str(config["data"].get("scaling"))
    desc += ("\n  Positive values only: "
             + str(config["data"].get("positive_values")))
    desc += ("\n  Group reweighting power: "
             + str(config["torch_params"].get("group_power")))
    desc += ("\n  Concept similarity transformation: "
             + config["labels"].get("transformation"))
    if estimator_type == "sklearn":
        desc += "\n  Model: " + config["model_name"]
    else:
        desc += "\n  Model: " + str(clf.model)
    if exploratory_comp == 1:
        desc += "\n  Parameters: " + str(param_grid)
        best_params = "NA"
    else:
        desc += "\n  Explored parameters: " + str(param_grid)
        desc += "\n  Best parameters: " + str(clf_grid.best_params_)
        best_params = str(clf_grid.best_params_)
    desc += "\n  " + perf

    with open(path + "description.txt", 'w', encoding='utf-8') as f:
        print(desc, file=f)

    # Append metrics to comparison file
    results_df = pd.DataFrame(
        data=[[
            ID,
            config["description"],
            str(datetime.datetime.now()),
            config["data"].get("scaling"),
            config["data"].get("positive_values"),
            config["torch_params"].get("group_power"),
            config["labels"].get("transformation"),
            config["labels"].get("threshold"),
            (config["model_name"] if estimator_type == "sklearn"
             else one_compact_line(clf.model)),
            str(param_grid),
            best_params,
            N,
            recall_train,
            recall_test,
            auc
        ]],
        columns=[
            "experiment",
            "description",
            "trained_on",
            "scaling",
            "positive_part",
            "collection_regul",
            "label_transformation",
            "threshold",
            "model",
            "explored_params",
            "best_params",
            "N",
            "recall_N_TRAIN",
            "recall_N_TEST",
            "AUC_TEST"
        ]
    )

    if os.path.isfile(results_file):
        previous_results = pd.read_csv(results_file, index_col=0)
        full_results = previous_results.append(results_df)
        full_results.index = list(range(len(full_results)))
        full_results.to_csv(results_file, header=True)
    else:
        results_df.to_csv(results_file, header=True)

    # Save per-label metrics
    size_train = len(X_train)
    size_test = len(X_test)
    n_labels = len(vocab_current)
    labels_in_test = pd.DataFrame([Y_test_orig.sum(axis=0)],
                                  columns=vocab_current)
    labels_in_train = pd.DataFrame([Y_train_orig.sum(axis=0)],
                                   columns=vocab_current)

    results = pd.DataFrame(
        columns=["ratio TRAIN",
                 "recall@10 TRAIN",
                 "ratio TEST",
                 "recall@10 TEST",
                 "AUC TEST"],
        index=labels_in_test.columns[labels_in_test.values[0] > 0]
    )

    for i, label in enumerate(vocab_current):
        if labels_in_test[label].values:
            mask_label = np.zeros(n_labels)
            mask_label[i] = 1
            mask_samples_train = Y_train_orig[:, i] > 0
            mask_samples_test = Y_test_orig[:, i] > 0
            if labels_in_train[label].values:

                results.loc[label] = [
                    labels_in_train[label].values[0] / size_train,
                    recall_n(Y_train_pred[mask_samples_train],
                             Y_train_orig[mask_samples_train] * mask_label,
                             n=10,
                             reduce_mean=True),
                    labels_in_test[label].values[0] / size_test,
                    recall_n(Y_test_pred[mask_samples_test],
                             Y_test_orig[mask_samples_test] * mask_label,
                             n=10,
                             reduce_mean=True),
                    roc_auc_score(Y_test_orig[:, i], Y_test_pred[:, i])
                    if (Y_test_orig[:, i].sum()) and (0 in Y_test_orig[:, i])
                    else np.nan
                ]
            else:
                results.loc[label] = [
                    0,
                    np.nan,
                    labels_in_test[label].values[0] / size_test,
                    recall_n(Y_test_pred[mask_samples_test],
                             Y_test_orig[mask_samples_test] * mask_label,
                             n=10,
                             reduce_mean=True)
                ]

    results.sort_values(by=['recall@10 TEST'], ascending=False, inplace=True)

    results.to_csv(per_label_results_file, header=True)

    if verbose:
        print("Detailed resuls & models in " + path)
        print("=== Finished ===")

    return clf


# ===================
# === MAIN SCRIPT ===
# ===================
def main():
    # --------------
    # --- CONFIG ---
    # --------------
    parser = argparse.ArgumentParser(
        description="A decoding experiment script for preprocessed fMRIs",
        epilog='''Examples:
        - python decoding_exp.py -C config_exp.json -g 0 -j 4 -v'''
    )

    parser.add_argument("-C", "--configuration",
                        default="spec_template.json",
                        help="Path of the JSON configuration file "
                             "or of a folder containing multiple configs")
    parser.add_argument("-M", "--mode",
                        default="train",  # TODO: implémenter modes
                        help="Execution mode for the experiment: "
                             "'train' or 'retrain' or 'evaluate'")
    parser.add_argument("-f", "--folder",
                        default="/home/parietal/rmenuet/work/cache/",
                        help="Path of the folder where to store "
                             "data, model and detailed results")
    parser.add_argument("-r", "--results_file",
                        default="../Data/results/decoding_perf.csv",
                        help="Path to the file where results are consolidated")
    parser.add_argument("-g", "--used_gpu",
                        type=int,
                        default=0,
                        help="GPU to be used (default 0, -1 to train on CPU)")
    parser.add_argument("-j", "--n_jobs",
                        type=int,
                        default=1,
                        help="Number of jobs "
                             "(parallel trainings during gridsearch-CV)")
    parser.add_argument("-H", "--HPC",
                        type=int,
                        default=0,
                        help="If > 0, launch the training on H nodes "
                             "on Slurm HPC using dask_jobqueue")
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Activates (many) outputs")
    parser.add_argument("-y", "--force",
                        action="store_true",
                        help="Forces retraining and overwriting "
                             "of already trained model")

    args = parser.parse_args()

    configurations = get_json_files_in_dir(args.configuration)

    if args.verbose:
        print(f"Configurations used:{configurations}")

    # ----------------------
    # --- RUN EXPERIMENT ---
    # ----------------------
    for conf in configurations:
        decoding_experiment(
            configuration=conf,
            mode=args.mode,
            folder=args.folder,
            results_file=args.results_file,
            used_gpu=args.used_gpu,
            n_jobs=args.n_jobs,
            verbose=args.verbose,
            force=args.force
        )


if __name__ == "__main__":
    # execute only if run as a script

    # fixing the seed and deterministic behavior
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # launch experiment
    main()
