import numpy as np
import torch
from numpy.random import choice
import muzero.utils as ut


class ReplayBuffer(object):
    def __init__(self, tdStep: int, maxGames: int = 10):
        """This class store all information relevant to replay a game."""
        self.history = {}
        self.tdStep = tdStep
        self.discount = 0.995
        self.maxGames = maxGames

    def add(self, step: dict, game: int):
        """Save one step of the game.

        step: A dictionary with keys:
                obs: Current game state
                act: Action taken. Int
                rew: Current reward received. Float
                pol: Current policy (torch.Tensor (1, actionSpace))
                val: Current estimated value for state. Float
        game: The index the game played.
        """
        if game in self.history:
            self.history[game].append(step)
        else:
            self.history[game] = [step]
        if len(self.history.keys()) > self.maxGames:
            k = list(self.history.keys()).pop(0)
            self.history.pop(k)

    def sample_batch(self, steps: int, batchSize: int = 32, game: int = None, prev_obs: int = 32) -> dict[str, torch.Tensor]:
        """Sample randomly a batch from a game.

        steps:     Number of steps to unroll the prediction.
        batchSize: Number of observations to stack.
        game:      The number of the game to sample the sequence of observations.
        """
        g = game
        if game is None:
            g = choice(list(self.history.keys()))
        selGame = self.history[g]
        pos = choice(len(selGame) - steps - prev_obs - 1, batchSize) + prev_obs
        batch = []
        for s in range(steps + 1):
            batch.append({
                "obs": torch.concat([self.make_obs(selGame, p+s, prev_obs) for p in pos], 0),
                "act": [selGame[p+s]["act"] for p in pos],
                "rew": torch.Tensor([selGame[p+s]["rew"] for p in pos]),
                "pol": torch.stack([selGame[p+s]["pol"] for p in pos]),
                "val": torch.Tensor([self.make_value(selGame, p + s) for p in pos])
            })
        return batch

    def make_value(self, game: dict, pos: int):
        bix = pos + self.tdStep
        value = 0
        if bix < len(game):
            value = game[bix]["val"] * self.discount**self.tdStep
        for j, step in enumerate(game[pos:bix]):
            value += step["rew"]*self.discount**j
        return value

    def make_obs(self, game: dict, pos: int, prev_obs: int = 32):
        """Stack previous observations and actions."""
        obsT = np.moveaxis(game[pos]["obs"], -1, 0) / 255.
        res = torch.Tensor(obsT)
        actShape = obsT.shape[1:]
        for i in range(prev_obs):
            if i != 0:
                obsT = np.moveaxis(game[pos-i]["obs"], -1, 0) / 255.
                res = torch.concat([res, torch.Tensor(obsT)], 0)
            act = ut.action_to_plane(game[pos-i]["act"], actShape)
            res = torch.concat([res, act], 0)
        return res[None, :]
