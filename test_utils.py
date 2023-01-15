import utils as ut
import torch


def test_action_to_plane():
    res = ut.action_to_plane(1, [2, 2])
    assert torch.equal(res, torch.Tensor([[[0., 1], [0., 0.]]]))


def test_action_to_plane2():
    res = ut.action_to_plane([1, 2], [2, 2])
    expe = torch.Tensor([
        [[0., 1.], [0., 0.]],
        [[0., 0.], [1., 0.]]
    ])
    assert torch.equal(res, expe)
