from typing import Union
import torch
import torch.nn as nn


# TODO: Deal with stride != 1?
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
