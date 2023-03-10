import torch
import torch.nn as nn
from muzero.buffer import ReplayBuffer
from muzero.models import Representation, Dynamics, Prediction
import muzero.utils as ut
from numpy.random import choice, uniform, dirichlet
from math import log, sqrt
from typing import Union


class Muzero(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.priority_epsilon = 0.001
        self.device = 'cuda'
        self.support_size = config["support_size"]
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
        self.g = Dynamics(
            self.dynInp, config["dynamic_hidden_size"] + [self.support_size*2 + 1])
        self.predInp = [config["representation_outputs"]
                        [-1], self.dynInp[1], self.dynInp[2]]
        self.f = Prediction(
            self.predInp, config["prediction_hidden_size"] + [config["action_space"]], (self.support_size*2 + 1))
        self.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=config["lr_init"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
        )
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
            probs = nn.Softmax(dim=1)(probs)
            noise = dirichlet(
                [self.config["root_dirichlet_alpha"]] * self.config["action_space"])
            frac = self.config["mcts_root_exploration"]
            for i, (n, p) in enumerate(zip(noise, probs[0])):
                root.children[i] = ut.Node(p*(1-frac) + n*frac)

            for j in range(nSimul):
                history = [root]
                history_act = []
                node = history[0]
                depth = 0
                # While node is expanded, traverse the tree until reach a leaf node and expanded.

                while len(node.children) > 0 and depth < self.config["mcts_max_depth"]:
                    act, node = self.select_action(node)
                    history.append(node)
                    history_act.append(act)
                    depth += 1

                parent = history[-2]
                node.hiddenState, reward = self.dynamics_net(
                    parent.hiddenState, act)
                node.reward = ut.support_to_scalar(
                    reward, self.support_size)
                probs, value = self.f(node.hiddenState)
                probs = nn.Softmax(dim=1)(probs)
                value = ut.support_to_scalar(
                    value, self.support_size)

                for i, p in enumerate(probs[0]):
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

    def train_batch(self, buffer: ReplayBuffer) -> None:
        """Train the model.

        batch: A list of dicts, each dict represent a batch of action in step k
            batch[0] = {obs     : torch.Tensor size(batchSize, (channel + 1)*prevObs, H, W)
                        act     : list[int]    size(batchSize, 1)
                        rew     : torch.Tensor size(batchSize, 1)
                        val     : torch.Tensor size(batchSize, 1)
                        policy  : torch.Tensor size(batchSize, actionSpace)
                        }
        """
        batch = buffer.sample_batch(self.config["rollout_steps"],
                                    self.config["batch_size"],
                                    self.config["observation_history"])
        gameId = batch[0]["gameId"]
        N = len(buffer.history[gameId])
        s = self.h(batch[0]["obs"])
        policyLoss, valueLoss, rewardLoss = (0, 0, 0)
        consistencyLoss, totalLoss = (0, 0)
        priorities = [[]]*len(batch)
        for i, b in enumerate(batch):
            if i != 0:
                with torch.no_grad():
                    sp = self.h(b["obs"])
                consistencyLoss += float(ut.consist_loss_func(
                    s.reshape(1, -1), sp.reshape(1, -1)))
            s, r = self.dynamics_net(s, b["act"])
            s.register_hook(lambda grad: grad * .5)
            p, v = self.f(s)
            vS = ut.support_to_scalar(v, self.support_size).squeeze()
            priorities[i] = abs(vS - b["val"]) + self.priority_epsilon
            value = ut.scalar_to_support(
                b["val"][:, None], self.config["support_size"])
            reward = ut.scalar_to_support(
                b["rew"][:, None], self.config["support_size"])
            currentValueLoss = (-value.squeeze() *
                                torch.nn.LogSoftmax(dim=1)(v)).sum(1) * (1/(N*b["pri"]))
            currentRewardLoss = (-reward.squeeze() * nn.LogSoftmax(dim=1)
                                 (r)).sum(1) * (1/(N*b["pri"]))
            currentPolicyLoss = (-b["pol"] *
                                 nn.LogSoftmax(dim=1)(p)).sum(1) * (1/(N*b["pri"]))
            currentPolicyLoss.register_hook(lambda grad: grad / len(batch))
            currentValueLoss.register_hook(lambda grad: grad / len(batch))
            currentRewardLoss.register_hook(lambda grad: grad / len(batch))
            policyLoss += currentPolicyLoss.mean()
            valueLoss += currentValueLoss.mean()
            rewardLoss += currentRewardLoss.mean()
        buffer.update_priorities(batch, priorities)
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
                     p=policy.numpy())
        if uniform() < self.config["random_action_threshold"]:
            act = choice(self.config["action_space"])
            # If action was taken by random, policy should be modify?
            # policy = [1/self.config["action_space"]]*self.config["action_space"]
        return {"pol": policy, "val": val, "act": act}
