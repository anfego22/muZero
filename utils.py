import torch
import torch.nn.functional as F
from math import prod, log, sqrt


def action_to_plane(act: int, dim: list = [60, 60]):
    return F.one_hot(act, prod(dim)).view(dim)[None, 1]


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
