import torch
import gym
from muzero.muzero import Muzero
from muzero.buffer import ReplayBuffer
import muzero.utils as ut
from uuid import uuid4
import numpy as np
import pickle
from datetime import datetime

ROLLOUT_STEPS = 6
DEVICE = 'cuda'

MUZERO_DEFAULT = {
    "observation_dim": [3, 96, 96],
    "observation_history": 8,
    "representation_outputs": [4, 4, 2],
    "dynamic_hidden_size": [18, 20, 8],
    "support_size": 21,
    "prediction_hidden_size": [8, 11, 6],
    "mcts_root_exploration": 0.8,
    "root_dirichlet_alpha": 0.25,
    "mcts_max_depth": ROLLOUT_STEPS * 2,
    "mcts_discount_value": 0.8,
    "mcts_simulations": 50,
    "pUCT_score_c1": 1.25,
    "pUCT_score_c2": 19652,
    "weight_decay": 1e-4,
    "momentum": 0.9,
    "lr_init": 0.05,
    "lr_decay_rate": 0.1,
    "lr_decay_steps": 350,
    "random_action_threshold": 0.8,
}


class Game(object):
    RANDOM_ACTION_DECAY = 0.995

    def __init__(self, env, random_action: float = 0.8,
                 agent_file: str = "muzero", buffer_file: str = "buffer",
                 buffer_size: int = 10, td_steps: int = 3, root_noise: float = 0.8, prev_obs: int = 32):
        self.agent_file = agent_file
        self.buffer_file = buffer_file
        MUZERO_DEFAULT["action_space"] = env.action_space.n
        MUZERO_DEFAULT["observation_history"] = prev_obs
        self.load_assets(buffer_size, td_steps)
        obsShape = self.agent.config["observation_dim"][1:]
        self.env = gym.wrappers.ResizeObservation(env, obsShape)
        self.env.reset()
        self.done = False
        self.stackObs = []
        self.agent.config["random_action_threshold"] = random_action
        self.agent.config["mcts_root_exploration"] = root_noise
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

    def load_assets(self, buffer_size: int, td_steps: int):
        start_t = datetime.now()
        print("Loading assets")
        try:
            with open("assets/" + self.agent_file, "rb") as f:
                self.agent = pickle.load(f)
        except:
            print("No file found. Creating default muzero")
            self.agent = Muzero(MUZERO_DEFAULT)
        pT = sum([p.numel() for p in self.agent.parameters()])
        print(f"Total parameters {pT}")
        try:
            with open("assets/" + self.buffer_file, "rb") as f:
                self.buffer = pickle.load(f)
        except:
            self.buffer = ReplayBuffer(td_steps, buffer_size)
        end_t = datetime.now() - start_t
        print(f"Load assets at: {end_t.total_seconds()}")

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
                    "val": res["val"]}
            self.buffer.add(step, gameId)
            self.stackObs.pop(0)
            self.stackObs.append((obs, res["act"]))

        end_t = datetime.now() - start_t
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
            prev_obs = self.agent.config["observation_history"]
            b = self.buffer.sample_batch(ROLLOUT_STEPS, prev_obs=prev_obs)
            loss, pL, rL, vL = self.agent.train_batch(b)
            if i != 0 and i % 10 == 0:
                print(
                    f"Total loss: {loss} policy loss: {pL} reward loss {rL} value loss {vL} step: {i}")
                with open("assets/" + self.agent_file, "wb") as f:
                    pickle.dump(self.agent, f)
