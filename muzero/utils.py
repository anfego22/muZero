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
