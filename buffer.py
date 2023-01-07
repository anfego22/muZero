import torch
from numpy.random import choice


class ReplayBuffer(object):
    def __init__(self, maxGames: int = 10):
        """This class store all information relevant to replay a game."""
        self.history = {}
        self.maxGames = maxGames

    def add(self, step: dict, game: int):
        """Save one step of the game.

        step: A dictionary with keys:
                obs: The state of the game
                act: The action taken
                rew: THe reward received
        game: The number of the game played.
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
                "act": torch.stack([selGame[p+s]["act"] for p in pos]),
                "rew": torch.stack([selGame[p+s]["rew"] for p in pos]),
            })
        return batch
