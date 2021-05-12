import math

from torch import nn
from src.commons.config.config_bestfitting import *
from .hard_example import *
from .lovasz_losses import *
import torch.nn.functional as F
import torch
from typing import Optional


# inspired with https://github.com/snorkel-team/snorkel/blob/master/snorkel/classification/loss.py
def binary_cross_entropy_with_probs(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    if len(input.shape) == 1:
        input = input.reshape(-1, 1)
        target = target.reshape(-1, 1)
    num_points, num_classes = input.shape

    # Note that t.new_zeros, t.new_full put tensor on same device as t
    cum_losses = input.new_zeros(num_points)
    target_temp = input.new_full((num_points,), 1, dtype=torch.long)
    for y in range(num_classes):

        # modified from https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Loss.cpp#L214
        # log-sum-exp trick: http://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        max_val = (-input[:, y]).clamp(min=0)

        y_loss = (1 - target_temp) * input[:, y] + max_val + ((-max_val).exp() + (-input[:, y] - max_val).exp()).log()
        if weight is not None:
            y_loss = y_loss * weight[y]
        cum_losses += target[:, y].float() * y_loss

    if reduction == "none":
        return cum_losses
    elif reduction == "mean":
        return cum_losses.mean()
    elif reduction == "sum":
        return cum_losses.sum()
    else:
        raise ValueError("Keyword 'reduction' must be one of ['none', 'mean', 'sum']")


def binary_focal_with_probs(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
gamma: int = 2
) -> torch.Tensor:
    if len(input.shape) == 1:
        input = input.reshape(-1, 1)
        target = target.reshape(-1, 1)
    num_points, num_classes = input.shape

    # Note that t.new_zeros, t.new_full put tensor on same device as t
    cum_losses = input.new_zeros(num_points)
    target_temp = input.new_full((num_points,), 1, dtype=torch.long)
    for y in range(num_classes):

        # modified from https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Loss.cpp#L214
        # log-sum-exp trick: http://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        max_val = (-input[:, y]).clamp(min=0)

        y_loss = (1 - target_temp) * input[:, y] + max_val + ((-max_val).exp() + (-input[:, y] - max_val).exp()).log()

        invprobs = F.logsigmoid(-input[:, y] * (target_temp * 2.0 - 1.0))
        y_loss = (invprobs * gamma).exp() * y_loss

        if weight is not None:
            y_loss = y_loss * weight[y]
        cum_losses += target[:, y].float() * y_loss

    if reduction == "none":
        return cum_losses
    elif reduction == "mean":
        return cum_losses.mean()
    elif reduction == "sum":
        return cum_losses.sum()
    else:
        raise ValueError("Keyword 'reduction' must be one of ['none', 'mean', 'sum']")


class FocalLossSimple(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logit, target, epoch=0):
        target = target.float()
        pred_prob = F.sigmoid(logit)
        ce = F.binary_cross_entropy_with_logits(logit, target, reduction='none')

        p_t = (target * pred_prob) + (1 - target) * (1 - pred_prob)

        modulating_factor = torch.pow((1.0 - p_t), self.gamma)
        if self.alpha is not None:
            alpha_factor = target * self.alpha + (1 - target) * (1 - self.alpha)
        else:
            alpha_factor = 1
        loss = alpha_factor * modulating_factor * ce
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target, epoch=0):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + \
               ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size())==2:
            loss = loss.sum(dim=1)
        return loss.mean()


class HardLogLoss(nn.Module):
    def __init__(self, soft_labels=False, symmetric=False):
        super(HardLogLoss, self).__init__()
        self.bce_loss = binary_cross_entropy_with_probs if soft_labels else nn.BCEWithLogitsLoss()
        self.__classes_num = NUM_CLASSES
        self.soft_labels = soft_labels
        self.symmetric = symmetric

    def forward(self, logits, labels,epoch=0):
        labels = labels.float()
        loss=0
        for i in range(NUM_CLASSES):
            logit_ac=logits[:,i]
            label_ac=labels[:,i]
            logit_ac, label_ac = (get_hard_samples_soft_symmetric(logit_ac,label_ac) if self.soft_labels else
                                  (get_hard_samples_symmetric(logit_ac,label_ac) if self.symmetric else get_hard_samples(logit_ac,label_ac)))
            if len(label_ac):
                loss += self.bce_loss(logit_ac, label_ac)
        loss = loss/NUM_CLASSES
        return loss


class SoftCEHardLogLoss(nn.Module):
    def __init__(self, hard_loss_weight=0.5):
        super(SoftCEHardLogLoss, self).__init__()
        self.soft_ce = binary_cross_entropy_with_probs
        self.log_loss = HardLogLoss(soft_labels=True)
        self.hard_loss_weight = hard_loss_weight

    def forward(self, logit, labels, epoch='for compatibility'):
        labels = labels.float()
        ce_loss = self.soft_ce(logit, labels)
        log_loss = self.log_loss.forward(logit, labels)
        loss = ce_loss*(1 - self.hard_loss_weight) + log_loss*self.hard_loss_weight
        return loss


class SoftFocalDifficultLogLoss(nn.Module):
    def __init__(self, hard_loss_weight=0.5):
        super(SoftFocalDifficultLogLoss, self).__init__()
        self.soft_focal = binary_focal_with_probs
        self.log_loss = HardLogLoss(soft_labels=True)
        self.hard_loss_weight = hard_loss_weight

    def forward(self, logit, labels, epoch='for compatibility'):
        labels = labels.float()
        ce_loss = self.soft_focal(logit, labels)
        log_loss = self.log_loss.forward(logit, labels)
        loss = ce_loss * (1 - self.hard_loss_weight) + log_loss * self.hard_loss_weight
        return loss
        # return ce_loss


# https://github.com/bermanmaxim/LovaszSoftmax/tree/master/pytorch
def lovasz_hinge(logits, labels, ignore=None, per_class=True):
    """
    Binary Lovasz hinge loss
      logits: [B, C] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, C] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_class:
        loss = 0
        for i in range(NUM_CLASSES):
            logit_ac = logits[:, i]
            label_ac = labels[:, i]
            loss += lovasz_hinge_flat(logit_ac, label_ac)
        loss = loss / NUM_CLASSES
    else:
        logits = logits.view(-1)
        labels = labels.view(-1)
        loss = lovasz_hinge_flat(logits, labels)
    return loss

# https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69053
class SymmetricLovaszLoss(nn.Module):
    def __init__(self):
        super(SymmetricLovaszLoss, self).__init__()
        self.__classes_num = NUM_CLASSES

    def forward(self, logits, labels,epoch=0):
        labels = labels.float()
        loss=((lovasz_hinge(logits, labels)) + (lovasz_hinge(-logits, 1 - labels))) / 2
        return loss


class FocalSymmetricLovaszHardLogLoss(nn.Module):
    def __init__(self):
        super(FocalSymmetricLovaszHardLogLoss, self).__init__()
        self.focal_loss = FocalLoss()
        self.slov_loss = SymmetricLovaszLoss()
        self.log_loss = HardLogLoss()

    def forward(self, logit, labels,epoch=0):
        labels = labels.float()
        focal_loss = self.focal_loss.forward(logit, labels, epoch)
        slov_loss = self.slov_loss.forward(logit, labels, epoch)
        log_loss = self.log_loss.forward(logit, labels, epoch)
        loss = focal_loss*0.5 + slov_loss*0.5 +log_loss * 0.5
        return loss


class FocalSymmetricHardLogLoss(nn.Module):
    def __init__(self):
        super(FocalSymmetricHardLogLoss, self).__init__()
        self.focal_loss = FocalLoss()
        self.log_loss = HardLogLoss(soft_labels=True)

    def forward(self, logit, labels,epoch=0):
        labels = labels.float()
        focal_loss = self.focal_loss.forward(logit, labels, epoch)
        log_loss = self.log_loss.forward(logit, labels, epoch)
        loss = focal_loss + log_loss * 0.25
        return loss


class SoftFocalSymmetricHardLogLoss(nn.Module):
    def __init__(self):
        super(SoftFocalSymmetricHardLogLoss, self).__init__()
        self.focal_loss = FocalLoss()
        self.log_loss = HardLogLoss(soft_labels=True)

    def forward(self, logit, labels, epoch=0):
        labels = labels.float()
        focal_loss = self.focal_loss.forward(logit, labels, epoch)
        log_loss = self.log_loss.forward(logit, labels, epoch)
        loss = focal_loss * 0.5 + log_loss * 0.5
        return loss


class FocalSymmetricLovaszSymHardLogLoss(nn.Module):
    def __init__(self):
        super(FocalSymmetricLovaszSymHardLogLoss, self).__init__()
        self.focal_loss = FocalLoss()
        self.slov_loss = SymmetricLovaszLoss()
        self.log_loss = HardLogLoss(symmetric=True)

    def forward(self, logit, labels,epoch=0):
        labels = labels.float()
        focal_loss = self.focal_loss.forward(logit, labels, epoch)
        slov_loss = self.slov_loss.forward(logit, labels, epoch)
        log_loss = self.log_loss.forward(logit, labels, epoch)
        loss = focal_loss*0.5 + slov_loss*0.5 +log_loss * 0.5
        return loss

# https://github.com/ronghuaiyang/arcface-pytorch
class ArcFaceLoss(nn.modules.Module):
    def __init__(self,s=30.0,m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.classify_loss = nn.CrossEntropyLoss()
        self.s = s
        self.easy_margin = False
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels, epoch=0):
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        loss1 = self.classify_loss(output, labels)
        loss2 = self.classify_loss(cosine, labels)
        gamma=1
        loss=(loss1+gamma*loss2)/(1+gamma)
        return loss
