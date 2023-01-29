import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
from torch.optim import Adam
from muzero.models import Representation, Dynamics, Prediction
import muzero.utils as ut
from numpy.random import choice, uniform, dirichlet
from math import log, sqrt
from typing import Union


class Muzero(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.device = 'cuda'
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
        self.to(self.device)
        self.optimizer = Adam(self.parameters(
        ),  lr=config["adam_lr"], weight_decay=config["adam_weight_decay"])
        self.mcts_simulations = config["mcts_simulations"]
        self.scaler = ut.MinMaxReward()
        self.eval()

    def dynamics_net(self, obs: torch.Tensor, act: Union[int, list[int]]):
        act = ut.action_to_plane(
            act, dim=[1] + self.dynInp[1:]).to(self.device)
        dynInp = torch.cat([obs, act], 1)
        return self.g(dynInp)

    def puct_score(self, parent: ut.Node, node: ut.Node):
        visits = parent.countVisits
        pb_c = self.config["pUCT_score_c1"] + log(
            (visits + self.config["pUCT_score_c2"] + 1) / self.config["pUCT_score_c2"])
        pb_c *= node.prob*sqrt(visits) / (1 + node.countVisits)
        value = node.get_value()
        if node.countVisits > 0:
            value = node.reward + self.scaler.normalize(value)
        return pb_c + value

    def select_action(self, parent: ut.Node) -> tuple[int, ut.Node]:
        scores = [self.puct_score(parent, child)
                  for child in parent.children.values()]
        maxAct = [i for i, v in enumerate(scores) if v == max(scores)]
        action = choice(maxAct)
        return action, parent.children[action]

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
            probs, _ = self.f(root.hiddenState)
            noise = dirichlet(
                [self.config["root_dirichlet_alpha"]] * self.config["action_space"])
            frac = self.config["mcts_root_exploration"]
            for i, (n, p) in enumerate(zip(noise, probs[0])):
                p = p / sum(probs[0])
                root.children[i] = ut.Node(p*(1-frac) + n*frac)

            for j in range(nSimul):
                history = [root]
                node = history[0]
                depth = 0
                # While node is expanded, traverse the tree until reach a leaf node and expanded.

                while len(node.children) > 0 and depth < self.config["mcts_max_depth"]:
                    act, node = self.select_action(node)
                    history.append(node)
                    depth += 1

                parent = history[-2]
                node.hiddenState, reward = self.dynamics_net(
                    parent.hiddenState, act)
                node.reward = float(reward)
                probs, value = self.f(node.hiddenState)
                value = float(value)

                for i, p in enumerate(probs[0]):
                    p = p / sum(probs[0])
                    node.children[i] = ut.Node(p)

                for n in reversed(history):
                    n.countVisits += 1
                    n.totalValue += value
                    self.scaler.update_val(n.get_value())
                    value = n.reward + self.config["mcts_discount_value"]*value

            policy = [n.countVisits for n in root.children.values()]
            norm = sum(policy)
            policy = torch.Tensor([p / norm for p in policy])
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
        policyLoss, valueLoss, rewardLoss = (0, 0, 0)
        consistencyLoss = 0
        for i, b in enumerate(batch):
            if i != 0:
                with torch.no_grad():
                    sp = self.h(b["obs"])
                consistencyLoss += float(ut.consist_loss_func(
                    s.reshape(1, -1), sp.reshape(1, -1)))
            s, r = self.dynamics_net(s, b["act"])
            p, v = self.f(s)
            policyLoss += loss1(p, b["pol"])
            valueLoss += loss2(v.squeeze(), b["val"])
            rewardLoss += loss2(r.squeeze(), b["rew"])
        totalLoss = policyLoss + valueLoss + rewardLoss + consistencyLoss
        self.optimizer.zero_grad()
        totalLoss.backward()
        self.optimizer.step()
        self.eval()
        return totalLoss, policyLoss, valueLoss, rewardLoss

    def act(self, obs: torch.Tensor, nSimul: int = None) -> dict:
        """Act according to policy."""
        if nSimul is None:
            nSimul = self.mcts_simulations
        policy, val = self.mcst(obs, nSimul)
        # act = [i for i, v in enumerate(policy) if v == max(policy)][0]
        act = choice(self.config["action_space"],
                     p=policy.numpy() / policy.numpy().sum())
        if uniform() < self.config["random_action_threshold"]:
            act = choice(self.config["action_space"])
            # If action was taken by random, policy should be modify?
            # policy = [1/self.config["action_space"]]*self.config["action_space"]
        return {"pol": policy, "val": val, "act": act}
