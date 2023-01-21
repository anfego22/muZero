import torch
from numpy.random import choice


class ReplayBuffer(object):
    def __init__(self, tdStep: int, maxGames: int = 10):
        """This class store all information relevant to replay a game."""
        self.history = {}
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

    def sample_batch(self, steps: int, batchSize: int = 32, game: int = None) -> dict[str, torch.Tensor]:
        """Sample randomly a batch from a game.

        steps:     Number of steps to unroll the prediction.
        batchSize: Number of observations to stack.
        game:      The number of the game to sample the sequence of observations.
        """
        nGames = len(self.history)
        g = game
        if game is None:
            g = choice(nGames)
        selGame = self.history[g]
        pos = choice(len(selGame) - steps, batchSize)
        batch = []
        for s in range(steps + 1):
            batch.append({
                "obs": torch.stack([selGame[p+s]["obs"] for p in pos]),
                "act": [selGame[p+s]["act"] for p in pos],
                "rew": torch.Tensor([selGame[p+s]["rew"] for p in pos]),
                "pol": torch.stack([selGame[p+s]["policy"] for p in pos]),
                "val": torch.Tensor([selGame[p+s]["value"] for p in pos]),
            })
        return batch
