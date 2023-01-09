import torch
from models import Representation, Dynamics, Prediction
import utils as ut
from numpy.random import choice
from math import log, sqrt


class Muzero():
    def __init__(self, config: dict):
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

    def dynamics_net(self, obs: torch.Tensor, act: int):
        act = ut.action_to_plane(act, dim=self.dynInp[1:])
        dynInp = torch.cat([obs, act], 1)
        return self.g(dynInp)

    def puct_score(self, parent: ut.Node, node: ut.Node):
        result = self.config["pUCT_score_c1"] + log(
            (parent.countVisits + self.config["pUCT_score_c2"] + 1) / self.config["pUCT_score_c2"])
        result *= node.prob*sqrt(parent.countVisits) / (1 + node.countVisits)
        result += node.get_value()
        return result

    def select_action(self, parent: ut.Node) -> tuple[int, ut.Node]:
        maxScore = -float("inf")
        for i, n in parent.children:
            score = self.puct_score(parent, n)
            if score > maxScore:
                res = (i, n)
        return res

    def mcst(self, obs: dict, nSimul: int = 50) -> float:
        """Run a monte carlo search tree.

        obs:    Current observation.
        nSimul: Number of times to run the MCST.

        Return

        Root node values.
        """
        root = ut.Node(0)
        root.hiddenState = self.h(obs)
        root.countVisits += 1
        probs, _ = self.f(root.hiddenState)
        for i, p in enumerate(probs):
            root.children[i] = ut.Node(p)

        for _ in range(nSimul):
            history = [root]
            node = history[0]
            # While node is expanded, traverse the tree until reach a leaf node and expanded.

            while len(node.children) > 0:
                act, node = self.select_action(node)
                history.append(node)

            parent = history[-2]
            node.hiddenState, node.reward = self.dynamics_net(
                parent.hiddenState, act)
            probs, value = self.f(node.hiddenState)

            for i, p in enumerate(probs):
                node.children[i] = ut.Node(p)

            for n in reversed(history):
                n.countVisits += 1
                n.totalValue += value
                value = n.reward + self.config["mcts_discount_value"]*value

        policy = [n.countVisits /
                  root.countVisits for n in root.children.values()]
        return policy, root.get_value()
