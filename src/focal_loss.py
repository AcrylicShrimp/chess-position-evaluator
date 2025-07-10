import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        ce_loss = F.nll_loss(log_probs, targets, reduction="none")

        pt = torch.exp(-ce_loss)

        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss

        if self.alpha is not None:
            self.alpha = self.alpha.to(targets.device)
            alpha_term = self.alpha.gather(0, targets)
            loss = alpha_term * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
