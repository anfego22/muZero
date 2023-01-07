from models import BasicBlock, DownSample
import torch


def test_basicblock_shape():
    bb = BasicBlock(3, 1)
    obs = torch.Tensor(1, 3, 8, 8)
    out = bb(obs)
    assert out.shape == (1, 3, 8, 8)


def test_downsample_shape():
    ds = DownSample(16, [8, 4])
    obs = torch.Tensor(1, 16, 8, 8)
    out = ds(obs)
    assert out.shape == (1, 4, 2, 2)
