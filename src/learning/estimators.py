import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from src.learning.models import ModelLinear


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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        train_dataset = DatasetFromNp(
            X_train,
            y_train,
            device=device,
        )
        val_dataset = DatasetFromNp(
            X_test,
            y_test,
            device=device,
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
        iterator = range(self.epochs)

        if (not self.batch_size) | (self.batch_size == -1):
            batch_size = len(X)
        else:
            batch_size = self.batch_size

        exp_lrs = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.8
        )

        sampler = None if sample_weights is None else WeightedRandomSampler(sample_weights, len(X))

        loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            shuffle=sampler is None,
                            num_workers=0,
                            # pin_memory=not self.sample_gpu,
                            timeout=120)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=0,
            # pin_memory=not self.sample_gpu,
            timeout=120,
        )

        for index, epoch in enumerate(iterator):
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

            current_val_loss = 0
            for X_tensor, y_tensor in val_loader:
                if self.gpu & (not self.sample_gpu):
                    # store data on GPU
                    X_tensor = X_tensor.to(device)
                    y_tensor = y_tensor.to(device)

                # Forward pass: Compute predicted y by passing x to the model
                y_pred = self.model(X_tensor)

                # Compute loss
                loss_val = self.loss_func(y_pred, y_tensor)
                loss_val = torch.mean(torch.sum(loss_val, 1))

                current_val_loss += loss_val
            # Save loss every *print_freq* epoch
            print(index, current_loss / len(X_train), current_val_loss / len(X_test))

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
