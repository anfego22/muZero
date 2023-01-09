from muzero import Muzero
import torch

config = {
    "observation_dim": [3, 64, 64],
    "observation_history": 3,
    "representation_outputs": [9, 4, 1],
    "dynamic_hidden_size": [16, 8],
    "prediction_hidden_size": [],
    "action_space": 3,
    "mcts_discount_value": 0.8,
    "pUCT_score_c1": 1.25,
    "pUCT_score_c2": 19652,
}

# TODO: Make it work with high and width that are multiple of two.


def test_representation_shapes():
    c = config.copy()
    mz = Muzero(c)
    obs = torch.Tensor(1, (3 + 1)*3, 64, 64)
    s = mz.h(obs)
    assert s.shape == (1, 1, 64 // (2**3), 64 // (2**3))


def test_dynamics_shapes():
    c = config.copy()
    mz = Muzero(c)
    dyn = torch.Tensor(1, 2, 64 // (2**3), 64 // (2**3))
    s1, r = mz.g(dyn)
    assert s1.shape == (1, 1, 64 // (2**3), 64 // (2**3))
    assert r.shape == (1, 1)


def test_prediction_shapes():
    c = config.copy()
    mz = Muzero(c)
    dyn = torch.Tensor(1, 1, 64 // (2**3), 64 // (2**3))
    p, v = mz.f(dyn)
    assert p.shape == (1, 3)
    assert v.shape == (1, 1)
