from unittest.mock import patch
from buffer import ReplayBuffer
import torch
import numpy as np


def test_buffer_add():
    rb = ReplayBuffer(2)
    rb.add({'obs': 0}, 0)
    rb.add({'obs': 1}, 1)
    assert rb.history[0][0]["obs"] == 0
    rb.add({'obs': 3}, 3)
    assert 0 not in rb.history
    assert rb.history[3][0]["obs"] == 3
    rb.add({'obs': 4}, 3)
    assert rb.history[3][1]["obs"] == 4


def test_buffer_add2():
    rb = ReplayBuffer(2)
    rb.add({'obs': 0}, 0)
    rb.add({'obs': 1}, 1)
    assert rb.history[0][0]["obs"] == 0
    rb.add({'obs': 3}, 3)
    assert 0 not in rb.history
    assert rb.history[3][0]["obs"] == 3
    rb.add({'obs': 0}, 0)
    assert 1 not in rb.history
    assert rb.history[0][0]["obs"] == 0


def test_batch():
    rb = ReplayBuffer(2)
    for i in range(10):
        step = {
            "obs": torch.Tensor([[i, 0], [i, 0]]),
            "act": i, "rew": i, "value": i,
            "policy": torch.Tensor([[0, 0, i]])
        }
        rb.add(step, 0)
    with patch('buffer.choice') as m:
        m.return_value = [0, 1, 6]
        res = rb.sample_batch(2, 3, 0)
    expectedObs0 = torch.stack([
        torch.Tensor([[0, 0], [0, 0]]),
        torch.Tensor([[1, 0], [1, 0]]),
        torch.Tensor([[6, 0], [6, 0]])
    ])
    assert torch.equal(res[0]["obs"], expectedObs0)
    expectedAct3 = [2, 3, 8]
    assert np.all(np.equal(res[2]["act"], expectedAct3))
    expectedRew2 = torch.Tensor([1, 2, 7])
    assert torch.equal(res[1]["rew"], expectedRew2)
