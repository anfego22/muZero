from typing import Union
import torch
import torch.nn as nn
from math import prod


class BasicBlock(nn.Module):
    def __init__(self, channel: int, stride: int = 1):
        """Basic block with skip connection."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channel, channel, stride=stride,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, stride=stride,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        out = self.layers(x)
        out = out + x
        out = nn.ReLU()(out)
        return out

# TODO: Add pooling


class DownSample(nn.Module):
    def __init__(self, inChannel: int, outChannel: Union[int, list]):
        """Reduction dimension layer."""
        super().__init__()
        layerList = []
        if not isinstance(outChannel, list):
            outChannel = [outChannel]
        for out in outChannel:
            layer = [nn.Conv2d(inChannel, out, kernel_size=3,
                               stride=2, padding=1, bias=False)]
            layer += [BasicBlock(out) for _ in range(2)]
            inChannel = out
            layerList += layer
        self.layer = nn.Sequential(*layerList)

    def forward(self, x: torch.Tensor):
        out = self.layer(x)
        return out


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
            linear += [nn.Linear(initDim, out), nn.ReLU(), nn.Dropout(.3)]
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
        outAct = nn.Softmax(1)
        layer = [BasicBlock(channel) for _ in range(2)]
        layer += [nn.Flatten(1)]
        if len(linOut) != 0:
            for out in linOut[:-1]:
                layer += [nn.Linear(initDim, out), nn.ReLU()]
                initDim = out
        self.layer = nn.Sequential(*layer)
        self.outLayer = [nn.Sequential(
            nn.Linear(initDim, linOut[-1]), outAct), nn.Linear(initDim, 1)]

    def forward(self, x) -> tuple:
        out = self.layer(x)
        policy, val = [layer(out) for layer in self.outLayer]
        return policy, val
