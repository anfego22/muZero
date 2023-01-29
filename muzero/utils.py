import torch
import torch.nn.functional as F
from math import prod, log, sqrt
from typing import Union


def action_to_plane(act: Union[int, list[int]], dim: list = [1, 60, 60]):
    if type(act) != list:
        act = [act]
    res = torch.stack(
        [torch.Tensor([a/18.]*prod(dim)).view(dim)
            for a in act]
    )
    return res


class Node(object):
    def __init__(self, prob: float):
        self.prob = prob
        self.countVisits = 0
        self.hiddenState = None
        self.children: dict[int, Node] = {}
        self.totalValue = 0
        self.reward = 0

    def get_value(self):
        if self.countVisits != 0:
            return self.totalValue / self.countVisits
        return 0


class MinMaxReward(object):
    def __init__(self, maxVal: float = None, minVal: float = None):
        self.minQ = float('inf') if not minVal else minVal
        self.maxQ = -float('inf') if not minVal else minVal

    def update_val(self, q: float) -> None:
        self.minQ = min(q, self.minQ)
        self.maxQ = max(q, self.minQ)

    def normalize(self, val: float) -> float:
        if self.minQ != self.maxQ:
            return (val - self.minQ) / (self.maxQ - self.minQ)
        return val


def consist_loss_func(f1, f2):
    """Consistency loss function: similarity loss
    Parameters
    """
    f1 = F.normalize(f1, p=2., dim=-1, eps=1e-5)
    f2 = F.normalize(f2, p=2., dim=-1, eps=1e-5)
    return -(f1 * f2).sum(dim=1)


# Shamefully copied from https://github.com/werner-duvaud/muzero-general/blob/master/models.py
# suport_to_scalar and scalar_to_support

def support_to_scalar(logits: torch.Tensor, support_size: int = 300):
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture
    """
    # Decode to a scalar
    probabilities = torch.softmax(logits, dim=1)
    support = (
        torch.tensor([x for x in range(-support_size, support_size + 1)])
        .expand(probabilities.shape)
        .float()
        .to(device=probabilities.device)
    )
    x = torch.sum(support * probabilities, dim=1, keepdim=True)

    # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (
        ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
        ** 2
        - 1
    )
    return x

# Shamefully copied from https://github.com/werner-duvaud/muzero-general/blob/master/models.py


def scalar_to_support(x: int, support_size: int = 300):
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture
    """
    # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

    # Encode on a vector
    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob = x - floor
    logits = torch.zeros(x.shape[0], x.shape[1],
                         2 * support_size + 1).to(x.device)
    logits.scatter_(
        2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
    )
    indexes = floor + support_size + 1
    prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits
