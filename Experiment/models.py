# IMPORTS
import torch
import torch.nn.functional as F


# MODELS
class ModelLinear(torch.nn.Module):
    """Basic linear model used for logistic regression"""

    def __init__(self,
                 n_feature=512, n_label=166):
        super().__init__()
        self.linear = torch.nn.Linear(n_feature, n_label)

    def forward(self, x):
        return self.linear(x)

    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))


class ModelLogReg(torch.nn.Module):

    def __init__(self,
                 n_feature, n_label, latent_dim=10,
                 input_dropout=0.0, hidden_dropout=0.3):
        super().__init__()
        self.input_dropout = torch.nn.Dropout(p=input_dropout)
        self.linear1 = torch.nn.Linear(n_feature, latent_dim)
        self.hidden_dropout = torch.nn.Dropout(p=hidden_dropout)
        self.linear2 = torch.nn.Linear(latent_dim, n_label)

    def forward(self, x):
        x = self.input_dropout(x)
        x = self.linear1(x)
        x = self.hidden_dropout(x)
        x = self.linear2(x)
        return x

    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))


class ModelLogReg1NonLin(torch.nn.Module):

    def __init__(self,
                 n_feature, n_label, latent_dim=100,
                 input_dropout=0.0, hidden_dropout=0.0):
        super().__init__()
        self.input_dropout = torch.nn.Dropout(p=input_dropout)
        self.hidden_dropout = torch.nn.Dropout(p=hidden_dropout)
        self.relu = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(n_feature, latent_dim)
        self.linear2 = torch.nn.Linear(latent_dim, n_label)

    def forward(self, x):
        x = self.input_dropout(x)
        x = self.linear1(x)
        x = self.hidden_dropout(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))


class ModelLogReg1NonLinBN(torch.nn.Module):

    def __init__(self,
                 n_feature, n_label, latent_dim=100):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(n_feature, latent_dim)
        self.bn1 = torch.nn.BatchNorm1d(latent_dim)
        self.linear2 = torch.nn.Linear(latent_dim, n_label)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))


class ModelLogReg3NonLin(torch.nn.Module):

    def __init__(self,
                 n_feature, n_label, latent_dim=100,
                 input_dropout=0.0, hidden_dropout=0.0):
        super(ModelLogReg3NonLin, self).__init__()
        self.input_dropout = torch.nn.Dropout(p=input_dropout)
        self.hidden_dropout = torch.nn.Dropout(p=hidden_dropout)
        self.relu = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(n_feature, latent_dim)
        self.linear2 = torch.nn.Linear(latent_dim, latent_dim)
        self.linear3 = torch.nn.Linear(latent_dim, latent_dim)
        self.linear4 = torch.nn.Linear(latent_dim, n_label)

    def forward(self, x):
        x = self.input_dropout(x)
        x = self.linear1(x)
        x = self.hidden_dropout(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.hidden_dropout(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.hidden_dropout(x)
        x = self.relu(x)
        x = self.linear4(x)
        return x

    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))


class ModelMultinomial(torch.nn.Module):

    def __init__(self,
                 n_feature, n_label, latent_dim=10,
                 input_dropout=0.0, hidden_dropout=0.3):
        super().__init__()
        self.input_dropout = torch.nn.Dropout(p=input_dropout)
        self.linear1 = torch.nn.Linear(n_feature, latent_dim)
        self.hidden_dropout = torch.nn.Dropout(p=hidden_dropout)
        self.linear2 = torch.nn.Linear(latent_dim, n_label)

    def forward(self, x):
        x = self.input_dropout(x)
        x = self.linear1(x)
        x = self.hidden_dropout(x)
        x = self.linear2(x)
        return x

    def predict_proba(self, x):
        return torch.softmax(self.forward(x), dim=1)


class ModelMultinomial1NonLin(torch.nn.Module):

    def __init__(self,
                 n_feature, n_label, latent_dim=100,
                 input_dropout=0.0, hidden_dropout=0.0):
        super().__init__()
        self.input_dropout = torch.nn.Dropout(p=input_dropout)
        self.hidden_dropout = torch.nn.Dropout(p=hidden_dropout)
        self.relu = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(n_feature, latent_dim)
        self.linear2 = torch.nn.Linear(latent_dim, n_label)

    def forward(self, x):
        x = self.input_dropout(x)
        x = self.linear1(x)
        x = self.hidden_dropout(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

    def predict_proba(self, x):
        return torch.softmax(self.forward(x), dim=1)


class ModelMultinomial1NonLinBN(torch.nn.Module):

    def __init__(self,
                 n_feature, n_label, latent_dim=100):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(n_feature, latent_dim)
        self.bn1 = torch.nn.BatchNorm1d(latent_dim)
        self.linear2 = torch.nn.Linear(latent_dim, n_label)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

    def predict_proba(self, x):
        return torch.softmax(self.forward(x), dim=1)


class ModelMultinomial3NonLin(torch.nn.Module):

    def __init__(self,
                 n_feature, n_label, latent_dim=100,
                 input_dropout=0.0, hidden_dropout=0.0):
        super().__init__()
        self.input_dropout = torch.nn.Dropout(p=input_dropout)
        self.hidden_dropout = torch.nn.Dropout(p=hidden_dropout)
        self.relu = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(n_feature, latent_dim)
        self.linear2 = torch.nn.Linear(latent_dim, latent_dim)
        self.linear3 = torch.nn.Linear(latent_dim, latent_dim)
        self.linear4 = torch.nn.Linear(latent_dim, n_label)

    def forward(self, x):
        x = self.input_dropout(x)
        x = self.linear1(x)
        x = self.hidden_dropout(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.hidden_dropout(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.hidden_dropout(x)
        x = self.relu(x)
        x = self.linear4(x)
        return x

    def predict_proba(self, x):
        return torch.softmax(self.forward(x), dim=1)


class ModelMultinomial3NonLinBN(torch.nn.Module):

    def __init__(self,
                 n_feature, n_label, latent_dim=100):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(n_feature, latent_dim)
        self.bn1 = torch.nn.BatchNorm1d(latent_dim)
        self.linear2 = torch.nn.Linear(latent_dim, latent_dim)
        self.bn2 = torch.nn.BatchNorm1d(latent_dim)
        self.linear3 = torch.nn.Linear(latent_dim, latent_dim)
        self.bn3 = torch.nn.BatchNorm1d(latent_dim)
        self.linear4 = torch.nn.Linear(latent_dim, n_label)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.linear4(x)
        return x

    def predict_proba(self, x):
        return torch.softmax(self.forward(x), dim=1)
