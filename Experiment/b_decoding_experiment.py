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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from sklearn import preprocessing

# ML
import torch
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

# custom modules
from a7_label import dumb_tagger
from src.utils.file import get_json_files_in_dir
from tools import (
    mask_rows,
    mkdir,
    yes_or_no,
    gridsearch_complexity,
    highly_corr_cols_np,
    one_compact_line,
    recall_n,
    mean_auc
)
from models import (
    ModelLinear,
    ModelLogReg,
    ModelLogReg1NonLin,
    ModelLogReg3NonLin,
    ModelLogReg1NonLinBN,
    ModelMultinomial,
    ModelMultinomial1NonLin,
    ModelMultinomial1NonLinBN,
    ModelMultinomial3NonLin,
    ModelMultinomial3NonLinBN
)


# ==============
# === LOSSES ===
# ==============
def continuous_cross_entropy_with_logits(pred, soft_targets, tol=1e-6):
    return (
        - torch.round(soft_targets) * soft_targets
        * torch.log(torch.clamp(torch.sigmoid(pred), tol, 1 - tol))
        - torch.round(1 - soft_targets) * (1 - soft_targets)
        * torch.log(torch.clamp(1 - torch.sigmoid(pred), tol, 1 - tol))
    )


def continuous_cross_entropy(pred, soft_targets):
    return torch.mean(
        torch.sum(
            continuous_cross_entropy_with_logits(pred, soft_targets),
            dim=1
        )
    )


def multinomial_cross_entropy(pred, soft_targets):
    return - soft_targets * torch.log_softmax(pred, dim=1)


# ================
# === DATASETs ===
# ================
class DatasetFromNp(Dataset):
    def __init__(self,
                 X, y,
                 device=torch.device("cpu")):
        self.X = torch.from_numpy(X).float().to(device)
        self.y = torch.from_numpy(y).float().to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return (
            self.X[index],
            self.y[index]
        )


# ==============
# === MODELS ===
# ==============
class PytorchEstimator(BaseEstimator):
    """
    A simple multi-labels logistic/multinomial regression model in Pytorch
    wrapped as a scikit-learn estimator for an easier implementation
    of cross-validations and pipelines.

    Training is by default done on GPU (if available)
    but prediction and scoring are done on CPU.

    Contrary to scikit-learn's logistic regression:
    - even if you only have one label, the target (provided for training
      and returned for prediction) is expected to be a 2D array
      (even if there is only 1 considered class
      it cannot be of shape (n,) it must be (n,1))
    - this implementation does not enforce binary labels/classes
      as ground truth at train time: continuous probabilities (between 0 - 1)
      can be provided as target of the fit function
    - predictions are also returned as continuous values between 0 and 1
      and should be thresholded for any exact classification or labelling task
    """

    def __init__(self,
                 model_class=ModelLinear,
                 loss_func=torch.nn.BCELoss(reduction='none'),
                 scoring_func=None,
                 epochs=1000, batch_size=-1,
                 adam=False,
                 lr=1e-1, momentum=0.9,
                 l1_reg=0, l2_reg=0,
                 weighted_samples=False,
                 gpu=True, used_gpu=0, sample_gpu=False,
                 verbose=0,
                 **kwargs):
        """
        The model is not instanciated there, its layers' dimensions
        will be infered from X and y shapes when using the 'fit' function

        Parameters
        ----------
        :param model_class: class inheriting torch.nn.module, default linear
            class that will be used to instanciate model
            its constructor should have at least the following arguments:
            - 'n_feature': the number of input features
            - 'n_label': the nummber of labels
            it also needs the following methods:
            - 'forward(input)': returns the output
              on which to apply the chosen loss
            - 'predict_likelihood(input)': returns the likelihood
              of each label for samples
        :param loss_func: a PyTorch loss function
            the loss function to be used for training and scoring:
            - should accept at least 3 PyTorch.tensor arguments:
              - the prediction of the model,
              - the ground truth,
              - an optional 'weight' kwarg
            - should return a differentiable tensor
              that can be backpropagated with .backward()
        :param scoring_func: None or a function
            if func: takes the same arguments loss_func, but as numpy.array
            if None: the opposite of the loss is used
        :param epochs: int, default=1000
            number of epochs for training
        :param batch_size: int, default=-1
            number of samples in each batch, if -1: the whole dataset is used
        :param adam: boolean, default=False
            whether to use the Adam optimizer
            if False, SGD is used
        :param lr: float, default=0.1
            learning rate for training
        :param momentum: float, default=0.9
            momentum used by the SGD optimizer (if used)
        :param l1_reg: float
            L1 regularization
        :param l2_reg: float
            L2 regularization
        :param weighted_samples: boolean, default=False
            if True: 1st dim of X corresponds to sampling weight
            and sampling is activated
        :param gpu: boolean, default=True
            whether the model should be trained using GPU,
            if False or if no GPU is available: the CPU is used
        :param used_gpu: int, default=0
            which GPU should be used (if gpu=True)
        :param sample_gpu: boolean, default=False
            whether to sample mini-batches directly on GPU or on CPU (before
            transfering to GPU), empirically: strangely sampling on CPU and then
                transfering to GPU seems quicker whenever the GPU is under heavy
                load sampling on GPU might be relevant for big mini-batches
        :param verbose: int, default=0
            verbosity level (mainly used at train time)
            0: nothing is printed
            1: final loss is printed at the end of training
            2: model parameters are displayed,
                training progression is displayed,
                loss curve is plotted
        :param kwargs: any list of named arguments
            is used to specify the model hyperparameters (architecture...)
            when it is instanciated during execution of the fit() method
        """
        super().__init__()

        self.model_class = model_class
        self.loss_func = loss_func
        self.scoring_func = scoring_func
        self.epochs = epochs
        self.batch_size = batch_size
        self.adam = adam
        self.lr = lr
        self.momentum = momentum
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.weighted_samples = weighted_samples
        self.gpu = gpu
        self.used_gpu = used_gpu
        self.sample_gpu = sample_gpu
        self.verbose = verbose

        if kwargs:
            # additional arguments keys stored for use within fit()
            self.additional_args = list(kwargs)
            # additional arguments stored as properties for cross_val
            self.__dict__.update(kwargs)
        else:
            self.additional_args = []

        if verbose > 1:
            print("Model will be instanciated using the following arguments:",
                  self.__dict__)

    @classmethod
    def from_file(cls, model_file, **kwargs):
        # Build model estimator from arguments
        model_estimator = cls(**kwargs)

        # Extract arguments required to build the model
        args = {}
        for key in model_estimator.additional_args:
            args[key] = model_estimator.__dict__[key]

        # Build the model
        model_estimator.model = torch.load(model_file)

        # Load the parameters from the saved file
        # model_estimator.model.load_state_dict(torch.load(model_file))

        return model_estimator

    def _get_param_names(self):
        """
        Overrides the default class method so that additional
        kwargs parameters are taken into account
        (the default class method looks for the model parameters
        in its class constructor signature, here we look among the
        properties of the object)
        """
        return sorted([p
                       for p in self.__dict__
                       if p != 'additional_args'])

    @staticmethod
    def check_X_y_weights(X, y=None, sample_weights=None):
        """
        Check that X, y (and sample_weights if provided) have the proper sizes
        and values
        """
        # X checking
        try:
            assert np.isfinite(np.max(X)), "X should only contain finite " \
                                           "numerical values"
        except Exception as e:
            print(str(e))
            raise RuntimeError("X should be a numerical array")

        # y checking
        if y is not None:
            try:
                assert not np.isnan(np.min(y))
            except Exception as e:
                print(str(e))
                raise RuntimeError(
                    "y should not contain NaN values"
                )

            try:
                assert (np.ndim(X) == np.ndim(y) == 2), "X and y should be " \
                                                        "2-dim arrays"
                assert (len(X) == len(y)), "X and y should have the same " \
                                           "'n_sample' first dimension"
            except Exception as e:
                print(str(e))
                raise RuntimeError(
                    "y should be a numerical array of the same size than X"
                )

            try:
                assert ((np.min(y) >= 0)
                        and
                        (np.max(y) <= 1)), "y values should be between 0 and 1"
                return y
            except Exception as e:
                # rounding issues might produce values outside of the
                # [0,1] range that we must correct
                y_corrected = y.copy()
                y_corrected[y < 0] = 0
                y_corrected[y > 1] = 1
                print("y should contain only values between 0 and 1")
                print(str(e))

                # returning the corrected y (bounded between 0 and 1)
                return y_corrected

        # sample_weights checking
        if sample_weights is not None:
            try:
                assert (np.min(sample_weights) >= 0), "sample weights should" \
                                                      " be positive values"
                assert (len(sample_weights) == len(X)), "there should be" \
                                                        " exactly one weight" \
                                                        " per sample"
            except Exception as e:
                print(str(e))
                raise RuntimeError(
                    "sample_weights should be a numerical array "
                    "of size n_samples and containing positive values"
                )

    def check_model(self):
        try:
            getattr(self, "model")
            n_feature = list(self.model.state_dict().values())[0].shape[1]
            n_label = list(self.model.state_dict().values())[-1].shape[0]
            return n_feature, n_label
        except AttributeError:
            raise RuntimeError("You must train (fit) model first")

    def fit(self, X, y,
            sample_weights=None, weighted_samples=False,
            n_jobs=0):
        """
        This is where the model is created (as input/output dimensions were
        unknown before) and trained.
        Even if the model is trained on GPU, it is returned on CPU so as to save
        GPU memory.

        :param X: numerical array-like, shape = (n_samples, n_features)
            features to be used
        :param y: numerical array-like, shape = (n_samples, n_classes)
            target labels (probability between 0 & 1 for each class) to be used
        :param sample_weights: numerical array-like, shape = (n_samples)
            sample weights to be used at train time
        :param weighted_samples: boolean, default=False
            if True, the 1st column of X is expected to be the samples weights
            useful to maintain samples weighting during cross-validation
        """
        y = self.check_X_y_weights(X, y, sample_weights)
        _, self.n_label = y.shape

        #
        # === Weights building ===
        #
        if weighted_samples:
            sample_weights = X[:, 0]
            X = X[:, 1:]
        self.n_sample, self.n_feature = X.shape

        #
        # === Model instanciation ===
        #
        args = {}
        for key in self.additional_args:
            args[key] = self.__dict__[key]

        model = self.model_class(n_feature=self.n_feature,
                                 n_label=self.n_label,
                                 **args)

        # === Device allocation ===
        cpu = torch.device("cpu")
        device = cpu
        if self.gpu:
            device = torch.device("cuda:" + str(self.used_gpu)
                                  if torch.cuda.is_available()
                                  else "cpu")

        self.model = model.to(device)

        # Dataset
        dataset = DatasetFromNp(
            X,
            y,
            device=device
        )

        #
        # === optimizer selection ===
        #
        if self.adam:
            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.lr)
        else:
            optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=self.lr,
                                        momentum=self.momentum)

        print_freq = max(self.epochs // 50, 1)
        losses_train = np.zeros(((self.epochs - 1) // print_freq) + 1)

        #
        # === Training ===
        #
        iterator = tqdm.trange(self.epochs)

        if (not self.batch_size) | (self.batch_size == -1):
            batch_size = len(X)
        else:
            batch_size = self.batch_size

        exp_lrs = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.8
        )

        if sample_weights is None:
            sampler = None
        else:
            sampler = WeightedRandomSampler(sample_weights,
                                            len(X))

        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            shuffle=sampler is None,
                            num_workers=0,
                            # pin_memory=not self.sample_gpu,
                            timeout=120)

        for epoch in iterator:
            # reset total loss for this epoch
            current_loss = 0

            # step for learning rate decrease
            exp_lrs.step()

            # run mini-batch
            for X_tensor, y_tensor in loader:
                # if the loader does not directly load on GPU
                if self.gpu & (not self.sample_gpu):
                    # store data on GPU
                    X_tensor = X_tensor.to(device)
                    y_tensor = y_tensor.to(device)

                # Forward pass: Compute predicted y by passing x to the model
                y_pred = self.model(X_tensor)

                # Compute loss
                loss_train = self.loss_func(y_pred, y_tensor)
                loss_train = torch.mean(torch.sum(loss_train, 1))

                current_loss += loss_train

                # compute regularizations
                if self.l1_reg:
                    for W in self.model.parameters():
                        loss_train += self.l1_reg * W.norm(1)

                if self.l2_reg:
                    for W in self.model.parameters():
                        loss_train += self.l2_reg * W.norm(2)

                # Backprop: zero gradients, backward pass, update weights
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

            # Save loss every *print_freq* epoch
            if self.verbose:

                if not epoch % print_freq:
                    losses_train[epoch // print_freq] = current_loss

        #
        # === Free (some) GPU memory ===
        #
        self.model = model.to(cpu)
        torch.cuda.empty_cache()

        #
        # === Display training perf ===
        #
        if self.verbose > 1:
            plt.plot(np.arange(0, self.epochs, print_freq),
                     losses_train,
                     label="Train")
            plt.title("Loss on TRAIN Dataset")
            plt.xlabel("Epochs")
            plt.ylabel("Loss (unregularized)")
            plt.legend()
            plt.show()
        if self.verbose:
            print("=> Final score:", self.score(X, y))

        return self

    def predict(self, X, y=None):
        """Prediction using the trained model (on CPU)"""
        self.check_model()
        self.check_X_y_weights(X)

        self.model.eval()
        y_pred = self.model.predict_proba(
            torch
            .from_numpy(X)
            .float()
            .to(torch.device("cpu"))
        ).detach().numpy()

        return y_pred

    def forward(self, X):
        """Prediction using the trained model (on CPU)"""
        self.check_model()
        self.check_X_y_weights(X)

        self.model.eval()
        y_pred = self.model.forward(
            torch
            .from_numpy(X)
            .float()
            .to(torch.device("cpu"))
        ).detach().numpy()

        return y_pred

    def score(self, X, y):
        """
        Scoring with the provided scoring function
        using the trained model (on CPU)
        """
        n_feature, _ = self.check_model()
        _, n_label = y.shape
        y = self.check_X_y_weights(X, y)

        if X.shape[1] == (n_feature + 1):
            X = X[:, 1:]

        assert (X.shape[1] == n_feature), "X is of the wrong shape"

        if self.scoring_func is None:
            y_pred = self.forward(X)

            loss = self.loss_func(torch.from_numpy(y_pred).float(),
                                  torch.from_numpy(y).float())
            loss = torch.mean(torch.sum(loss, 1)).numpy()

            return - loss
        else:
            y_pred = self.predict(X)
            return self.scoring_func(y_pred, y)

# ================================================
# === FUNCTION TO RUN FULL DECODING EXPERIMENT ===
# ================================================
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
        "logreg": ModelLogReg,
        "logreg_1nonlin": ModelLogReg1NonLin,
        "logreg_3nonlin": ModelLogReg3NonLin,
        "logreg_1nonlin_bn": ModelLogReg1NonLinBN,
        "multinomial": ModelMultinomial,
        "multinomial_1nonlin": ModelMultinomial1NonLin,
        "multinomial_1nonlin_bn": ModelMultinomial1NonLinBN,
        "multinomial_3nonlin": ModelMultinomial3NonLin,
        "multinomial_3nonlin_bn": ModelMultinomial3NonLinBN,
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
    # TODO: cleaner
    mask_test = (meta["collection_id"]
                 .isin(config["evaluation"]["test_IDs"]))
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
            model_class=model_classes[config["model_name"]],
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

    if args.verbose:
        print("\n>>> Verbosity turned on <<<\n")

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
