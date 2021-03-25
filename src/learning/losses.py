import torch


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

