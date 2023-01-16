import torch
import torch.nn.functional as F
from math import prod, log, sqrt
from typing import Union


def action_to_plane(act: Union[int, list[int]], dim: list = [60, 60]):
    if type(act) is list:
        res = torch.stack(
            [F.one_hot(torch.LongTensor([a]), prod(dim)).view(dim)[None, :]
             for a in act]
        )
        return res
    return F.one_hot(torch.LongTensor([act]), prod(dim)).view(dim)[None, :]


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
