# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.

import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch.nn as nn
from enum import IntEnum

def stable_kl(logit, target, epsilon=1e-6, reduce=True):
    logit = logit.view(-1, logit.size(-1)).float()
    target = target.view(-1, target.size(-1)).float()
    bs = logit.size(0)
    p = F.log_softmax(logit, 1).exp()
    y = F.log_softmax(target, 1).exp()
    rp = -(1.0/(p + epsilon) -1 + epsilon).detach().log()
    ry = -(1.0/(y + epsilon) -1 + epsilon).detach().log()
    if reduce:
        return (p* (rp- ry) * 2).sum() / bs
    else:
        return (p* (rp- ry) * 2).sum()

def entropy(logit, epsilon=1e-6, reduce=True):
    logit = logit.view(-1, logit.size(-1)).float()

    bs = logit.size(0)
    p = F.log_softmax(logit, 1).exp()
    log_p = p.log()
    loss = - p * log_p

    if reduce:
        return loss.sum() / bs
    else:
        return loss.sum()

class Criterion(_Loss):
    def __init__(self, alpha=1.0, name='criterion'):
        super().__init__()
        """Alpha is used to weight each loss term
        """
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """weight: sample weight
        """
        return

class CeCriterion(Criterion):
    def __init__(self, alpha=1.0, name='Cross Entropy Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """weight: sample weight
        """
        if weight:
            loss = torch.mean(F.cross_entropy(input, target, reduce=False, ignore_index=ignore_index) * weight)
        else:
            loss = F.cross_entropy(input, target, ignore_index=ignore_index)
        loss = loss * self.alpha
        return loss


class SeqCeCriterion(CeCriterion):
    def __init__(self, alpha=1.0, name='Seq Cross Entropy Criterion'):
        super().__init__(alpha, name)

    def forward(self, input, target, weight=None, ignore_index=-1):
        target = target.view(-1)
        if weight:
            loss = torch.mean(F.cross_entropy(input, target, reduce=False, ignore_index=ignore_index) * weight)
        else:
            loss = F.cross_entropy(input, target, ignore_index=ignore_index)
        loss = loss * self.alpha
        return loss

class MseCriterion(Criterion):
    def __init__(self, alpha=1.0, name='MSE Regression Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """weight: sample weight
        """
        if weight:
            loss = torch.mean(F.mse_loss(input.squeeze(), target, reduce=False) * 
                              weight.reshape((target.shape[0], 1)))
        else:
            loss = F.mse_loss(input.squeeze(), target)
        loss = loss * self.alpha
        return loss

class KlCriterion(Criterion):
    def __init__(self, alpha=1.0, name='KL Div Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1, target_soft=False):
        """input/target: logits
        """
        input = input.float()
        target = target.float()
        if not target_soft:
           loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32), F.softmax(target, dim=-1, dtype=torch.float32), reduction='batchmean')
        else:
            loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32),
                            target, reduction='batchmean')
        loss = loss * self.alpha
        return loss

class NsKlCriterion(Criterion):
    def __init__(self, alpha=1.0, name='KL Div Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """input/target: logits
        """
        input = input.float()
        target = target.float()
        loss = stable_kl(input, target.detach()) 
        loss = loss * self.alpha
        return loss


class SymKlCriterion(Criterion):
    def __init__(self, alpha=1.0, name='KL Div Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1, reduction='batchmean'):
        """input/target: logits
        """
        input = input.float()
        target = target.float()
        loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32), F.softmax(target.detach(), dim=-1, dtype=torch.float32), reduction=reduction) + \
            F.kl_div(F.log_softmax(target, dim=-1, dtype=torch.float32), F.softmax(input.detach(), dim=-1, dtype=torch.float32), reduction=reduction)
        loss = loss * self.alpha
        return loss

class NsSymKlCriterion(Criterion):
    def __init__(self, alpha=1.0, name='KL Div Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """input/target: logits
        """
        input = input.float()
        target = target.float()
        loss = stable_kl(input, target.detach()) + \
                stable_kl(target, input.detach())
        loss = loss * self.alpha
        return loss

class JSCriterion(Criterion):
    def __init__(self, alpha=1.0, name='JS Div Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1, reduction='batchmean'):
        """input/target: logits
        """
        input = input.float()
        target = target.float()
        m = F.softmax(target.detach(), dim=-1, dtype=torch.float32) + \
            F.softmax(input.detach(), dim=-1, dtype=torch.float32)
        m = 0.5 * m
        loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32), m, reduction=reduction) + \
            F.kl_div(F.log_softmax(target, dim=-1, dtype=torch.float32), m, reduction=reduction)
        loss = loss * self.alpha
        return loss

class HLCriterion(Criterion):
    def __init__(self, alpha=1.0, name='Hellinger Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1, reduction='batchmean'):
        """input/target: logits
        """
        input = input.float()
        target = target.float()
        si = F.softmax(target.detach(), dim=-1, dtype=torch.float32).sqrt_()
        st = F.softmax(input.detach(), dim=-1, dtype=torch.float32).sqrt_()
        loss = F.mse_loss(si, st)
        loss = loss * self.alpha
        return loss


class RankCeCriterion(Criterion):
    def __init__(self, alpha=1.0, name='Cross Entropy Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1, pairwise_size=1):
        input = input.view(-1, pairwise_size)
        target = target.contiguous().view(-1, pairwise_size)[:, 0]
        if weight:
            loss = torch.mean(F.cross_entropy(input, target, reduce=False, ignore_index=ignore_index) * weight)
        else:
            loss = F.cross_entropy(input, target, ignore_index=ignore_index)
        loss = loss * self.alpha
        return loss

class SpanCeCriterion(Criterion):
    def __init__(self, alpha=1.0, name='Span Cross Entropy Criterion'):
        super().__init__()
        """This is for extractive MRC, e.g., SQuAD, ReCoRD ... etc
        """
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """weight: sample weight
        """
        assert len(input) == 2
        start_input, end_input = input
        start_target, end_target = target
        if weight:
            b = torch.mean(F.cross_entropy(start_input, start_target, reduce=False, ignore_index=ignore_index) * weight)
            e = torch.mean(F.cross_entropy(end_input, end_target, reduce=False, ignore_index=ignore_index) * weight)
        else:
            b = F.cross_entropy(start_input, start_target, ignore_index=ignore_index)
            e = F.cross_entropy(end_input, end_target, ignore_index=ignore_index)
        loss = 0.5 * (b + e) * self.alpha
        return loss

class MlmCriterion(Criterion):
    def __init__(self, alpha=1.0, name='BERT pre-train Criterion'):
        super().__init__()
        self.alpha = alpha
        self.name = name

    def forward(self, input, target, weight=None, ignore_index=-1):
        """TODO: support sample weight, xiaodl
        """
        mlm_y, y = target
        mlm_p, nsp_p = input
        mlm_p = mlm_p.view(-1, mlm_p.size(-1))
        mlm_y = mlm_y.view(-1)
        mlm_loss = F.cross_entropy(mlm_p, mlm_y, ignore_index=ignore_index)
        nsp_loss = F.cross_entropy(nsp_p, y)
        loss = mlm_loss + nsp_loss
        loss = loss * self.alpha
        return loss

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0, metric = 'cos'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.metric = metric
        self.sim = nn.CosineSimilarity(dim=-1)
        self.loss =  torch.nn.BCEWithLogitsLoss()
        # print('ContrastiveLoss, Metric:', self.metric)

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def get_loss(self, x0, x1, y):
        # import pdb
        # pdb.set_trace()
        #elf.check_type_forward((x0, x1, y))

        # euclidian distance
        if self.metric == 'l2':
            diff = x0 - x1
            dist_sq = torch.sum(torch.pow(diff, 2), 1) / x0.shape[-1]
            dist = torch.sqrt(dist_sq)
        elif self.metric == 'cos':
            sim = self.sim(x0, x1) / 0.1
            #print(x0, x1, torch.sum(torch.pow(x0-x1, 2), 1) / x0.shape[-1], dist, dist_sq)
        else:
            print("Error Loss Metric!!")
            return 0
        #dist = torch.sum( - x0 * x1 / np.sqrt(x0.shape[-1]), 1).exp()
        #dist_sq = dist ** 2

        # mdist = self.margin - dist
        # dist = torch.clamp(mdist, min=0.0)
        # loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        # loss = torch.sum(loss) / 2.0 / x0.size()[0]
        loss = self.loss(sim, y)
        return loss#, dist_sq, dist

    def forward(self, x0, x1, y):
        loss1 = self.get_loss(x0, x1.detach(), y)
        loss2  = self.get_loss(x1.detach(), x0, y)
        return (loss1 + loss2) /2



class LossCriterion(IntEnum):
    CeCriterion = 0
    MseCriterion = 1
    RankCeCriterion = 2
    SpanCeCriterion = 3
    SeqCeCriterion = 4
    MlmCriterion = 5
    KlCriterion = 6
    SymKlCriterion = 7
    NsKlCriterion = 8
    NsSymKlCriterion = 9
    JSCriterion = 10
    HLCriterion = 11


LOSS_REGISTRY = {
     LossCriterion.CeCriterion: CeCriterion,
     LossCriterion.MseCriterion: MseCriterion,
     LossCriterion.RankCeCriterion: RankCeCriterion,
     LossCriterion.SpanCeCriterion: SpanCeCriterion,
     LossCriterion.SeqCeCriterion: SeqCeCriterion,
     LossCriterion.MlmCriterion: MlmCriterion,
     LossCriterion.KlCriterion: KlCriterion,
     LossCriterion.SymKlCriterion: SymKlCriterion,
     LossCriterion.NsKlCriterion: NsKlCriterion,
     LossCriterion.NsSymKlCriterion: NsSymKlCriterion,
     LossCriterion.JSCriterion: JSCriterion,
     LossCriterion.HLCriterion: HLCriterion,
}
