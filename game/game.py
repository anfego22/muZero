import torch
import gym
from muzero.muzero import Muzero
from muzero.buffer import ReplayBuffer
import muzero.utils as ut
from uuid import uuid4
import numpy as np
import pickle
from datetime import datetime

DEVICE = 'cuda'


class Game(object):
    RANDOM_ACTION_DECAY = 0.995

    def __init__(self, env, agent_config: dict, agent_file: str = "muzero",
                 buffer_file: str = "buffer", buffer_size: int = 10):
        self.agent_file = agent_file
        self.buffer_file = buffer_file
        self.agent_config = agent_config
        self.load_assets(buffer_size)
        obsShape = self.agent.config["observation_dim"][1:]
        self.env = gym.wrappers.ResizeObservation(env, obsShape)
        self.env.reset()
        self.done = False
        self.stackObs = []
        self.agent.config = agent_config
        self.device = DEVICE

    def make_image(self) -> torch.Tensor:
        """Convert a list of tuples into a 3D tensor.

        stackObs: List of tuples with observation and actions.
        """
        obsT = np.moveaxis(self.stackObs[0][0], -1, 0) / 255.
        res = torch.Tensor(obsT)
        for i, (obs, act) in enumerate(self.stackObs):
            if i != 0:
                obsT = np.moveaxis(obs, -1, 0) / 255.
                res = torch.concat([res, torch.Tensor(obsT)], 0)
            act = ut.action_to_plane(act, obs.shape[:-1])
            res = torch.concat([res, act], 0)
        obs = res[None, :].to(self.device)
        return obs

    def reset_env(self):
        self.env.reset()
        self.done = False
        self.stackObs = []

    def load_assets(self, buffer_size: int):
        start_t = datetime.now()
        print("Loading assets")
        try:
            with open("assets/" + self.agent_file, "rb") as f:
                self.agent = pickle.load(f)
        except:
            print("No file found. Creating default muzero")
            self.agent = Muzero(self.agent_config)
        pT = sum([p.numel() for p in self.agent.parameters()])
        print(f"Total parameters {pT}")
        try:
            with open("assets/" + self.buffer_file, "rb") as f:
                self.buffer = pickle.load(f)
        except:
            self.buffer = ReplayBuffer(buffer_size)
        end_t = datetime.now() - start_t
        print(f"Load assets at: {end_t.total_seconds()}")

    def make_mc_returns(self, buffer: ReplayBuffer, gameId: str) -> None:
        v = 0
        for step in reversed(buffer.history[gameId]):
            step["val"] = step["val"] + v
            v = step["rew"] + self.agent.config["mcts_discount_value"]*v

    def play_game(self):
        gameId = uuid4().__str__()[:6]
        start_t = datetime.now()
        print(f"Playing game {gameId}")
        while not self.done:
            if len(self.stackObs) < self.agent.config["observation_history"]:
                obs, reward, self.done, _, info = self.env.step(0)
                self.stackObs.append((obs, 0))
                continue
            obsHistory = self.make_image()
            res = self.agent.act(obsHistory)
            obs, reward, self.done, _, _ = self.env.step(res["act"])
            step = {"obs": self.stackObs[-1][0],
                    "act": res["act"], "rew": reward, "pol": res["pol"],
                    "val": res["val"], "pri": 1}
            self.buffer.add(step, gameId)
            self.stackObs.pop(0)
            self.stackObs.append((obs, res["act"]))

        end_t = datetime.now() - start_t
        self.make_mc_returns(self.buffer, gameId)
        print(f"game {gameId} finish at: {end_t.total_seconds()}")
        start_t = datetime.now()
        with open("assets/" + self.buffer_file, "wb") as f:
            pickle.dump(self.buffer, f)
        end_t = datetime.now() - start_t
        print(f"Save game history at: {end_t.total_seconds()}")
        self.reset_env()
        self.agent.config["random_action_threshold"] *= self.RANDOM_ACTION_DECAY
        self.agent.config["mcts_root_exploration"] *= self.RANDOM_ACTION_DECAY

    def run(self, games):
        for g in range(games):
            self.play_game()

    def train_agent(self, epoch: int = 500):
        for i in range(epoch):
            loss, pL, rL, vL = self.agent.train_batch(self.buffer)
            if i != 0 and i % 10 == 0:
                print(
                    f"Total loss: {loss} policy loss: {pL} reward loss {rL} value loss {vL} step: {i}")
                with open("assets/" + self.agent_file, "wb") as f:
                    pickle.dump(self.agent, f)
