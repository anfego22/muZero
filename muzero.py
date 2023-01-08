import torch
import torch.nn as nn
from models import BasicBlock, DownSample
from typing import Union
from math import prod


class Representation(nn.Module):
    def __init__(self, chann: int, outDownSample: list):
        super().__init__()
        self.down = DownSample(chann, outDownSample)

    def forward(self, x):
        out = self.down(x)
        return out


class Dynamics(nn.Module):
    def __init__(self, inpDim: list, linOut: list = [1]):
        super().__init__()
        channel = inpDim[0]
        linOut += [1]
        initDim = prod([inpDim[0] - 1] + inpDim[1:])
        layer = [BasicBlock(channel) for _ in range(2)]
        layer += [nn.Conv2d(channel, channel - 1, 3, stride=1, padding=1)]
        self.layer = nn.Sequential(*layer)
        linear = []
        for out in linOut:
            linear += [nn.Linear(initDim, out), nn.Sigmoid(), nn.Dropout(.3)]
            initDim = out
        self.dense = nn.Sequential(*linear)

    def forward(self, x):
        h = self.layer(x)
        h1 = nn.Flatten(1)(h)
        r = self.dense(h1)
        return h, r


class Prediction(nn.Module):
    def __init__(self, inpDim: list, linOut: list[int]):
        super().__init__()
        channel = inpDim[0]
        initDim = prod(inpDim)
        layer = [BasicBlock(channel) for _ in range(2)]
        layer += [nn.Flatten(1)]
        if len(linOut) > 1:
            for out in linOut[:-1]:
                layer += [nn.Linear(initDim, out)]
                initDim = out
        self.layer = nn.Sequential(*layer)
        self.outLayer = [nn.Linear(initDim, linOut[-1]), nn.Linear(initDim, 1)]

    def forward(self, x) -> tuple:
        out = self.layer(x)
        policy, val = [layer(out) for layer in self.outLayer]
        return policy, val


class Muzero():
    def __init__(self, config: dict):
        self.config = config
        self.repInp = (config["observation_dim"][0] + 1) * \
            config["observation_history"]
        self.h = Representation(
            self.repInp, config["representation_outputs"])
        self.dynInp = [
            config["representation_outputs"][-1] + 1,
            config["observation_dim"][1] // (2 **
                                             len(config["representation_outputs"])),
            config["observation_dim"][2] // (2 **
                                             len(config["representation_outputs"]))
        ]
        self.g = Dynamics(self.dynInp, config["dynamic_hidden_size"])
        self.predInp = [config["representation_outputs"]
                        [-1], self.dynInp[1], self.dynInp[2]]
        self.f = Prediction(
            self.predInp, config["prediction_hidden_size"] + [config["action_space"]])
