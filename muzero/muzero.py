import torch
import torch.nn as nn
from torch.optim import Adam
from muzero.models import Representation, Dynamics, Prediction
import muzero.utils as ut
from numpy.random import choice
from math import log, sqrt
from typing import Union


class Muzero(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.repInp = (config["observation_dim"][0] + 1) * \
            config["observation_history"]
        self.h = Representation(
            self.repInp, config["representation_outputs"])
        self.dynInp = [
            config["representation_outputs"][-1] + 1,
            config["observation_dim"][1] // (2 **
                                             len(config["representation_outputs"])),
            config["observation_dim"][2] // (2 **
                                             len(config["representation_outputs"]))
        ]
        self.g = Dynamics(self.dynInp, config["dynamic_hidden_size"])
        self.predInp = [config["representation_outputs"]
                        [-1], self.dynInp[1], self.dynInp[2]]
        self.f = Prediction(
            self.predInp, config["prediction_hidden_size"] + [config["action_space"]])
        self.optimizer = Adam(self.parameters(
        ),  lr=config["adam_lr"], weight_decay=config["adam_weight_decay"])
        self.mcts_simulations = config["mcts_simulations"]
        self.eval()

    def dynamics_net(self, obs: torch.Tensor, act: Union[int, list[int]]):
        act = ut.action_to_plane(act, dim=self.dynInp[1:])[None, :]
        dynInp = torch.cat([obs, act], 1)
        return self.g(dynInp)

    def puct_score(self, parent: ut.Node, node: ut.Node):
        visits = sum([n.countVisits for n in parent.children.values()])
        result = self.config["pUCT_score_c1"] + log(
            (visits + self.config["pUCT_score_c2"] + 1) / self.config["pUCT_score_c2"])
        result *= node.prob*sqrt(visits) / (1 + node.countVisits)
        result += node.get_value()
        return result

    def select_action(self, parent: ut.Node) -> tuple[int, ut.Node]:
        maxScore = -float("inf")
        for i, n in parent.children.items():
            score = self.puct_score(parent, n)
            if score > maxScore:
                maxScore = score
                res = (i, n)
        return res

    def mcst(self, obs: dict, nSimul: int = 50) -> tuple[torch.Tensor, float]:
        """Run a monte carlo search tree.

        obs:    Current observation.
        nSimul: Number of times to run the MCST.

        Return

        Root node values.
        """
        with torch.no_grad():
            root = ut.Node(0)
            root.hiddenState = self.h(obs)
            root.countVisits += 1
            probs, _ = self.f(root.hiddenState)
            for i, p in enumerate(probs[0]):
                root.children[i] = ut.Node(p)

            for j in range(nSimul):
                history = [root]
                node = history[0]
                # While node is expanded, traverse the tree until reach a leaf node and expanded.

                while len(node.children) > 0:
                    act, node = self.select_action(node)
                    history.append(node)

                parent = history[-2]
                node.hiddenState, reward = self.dynamics_net(
                    parent.hiddenState, act)
                node.reward = float(reward)
                probs, value = self.f(node.hiddenState)
                value = float(value)

                for i, p in enumerate(probs[0]):
                    node.children[i] = ut.Node(p)

                for n in reversed(history):
                    n.countVisits += 1
                    n.totalValue += value
                    value = n.reward + self.config["mcts_discount_value"]*value

            policy = [n.countVisits for n in root.children.values()]
            norm = sum(policy)
            policy = [p / norm for p in policy]
            return policy, root.get_value()

    def train_batch(self, batch: dict) -> None:
        """Train the model.

        batch: A list of dicts, each dict represent a batch of action in step k
            batch[0] = {obs     : torch.Tensor size(batchSize, (channel + 1)*prevObs, H, W)
                        act     : list[int]    size(batchSize, 1)
                        rew     : torch.Tensor size(batchSize, 1)
                        val     : torch.Tensor size(batchSize, 1)
                        policy  : torch.Tensor size(batchSize, actionSpace)
                        }
        """
        s = self.h(batch[0]["obs"])
        loss1, loss2 = nn.BCEWithLogitsLoss(), nn.MSELoss()
        totalLoss = 0
        for b in batch:
            s, r = self.dynamics_net(s, b["act"])
            p, v = self.f(s)
            totalLoss += loss1(p, b["pol"])
            totalLoss += loss2(v, b["val"])
            totalLoss += loss2(r, b["rew"])
        self.optimizer.zero_grad()
        totalLoss.backward()
        self.optimizer.step()
        self.eval()

    def act(self, obs: torch.Tensor, nSimul: int = None) -> dict:
        """Act according to policy."""
        if nSimul is None:
            nSimul = self.mcts_simulations
        policy, val = self.mcst(obs, nSimul)
        return {"pol": policy, "val": val}
