import argparse
import gym
from game.game import Game
import pickle

ROLLOUT_STEPS = 6
RANDOM_ACTION_DECAY = 0.995


parser = argparse.ArgumentParser()
parser.add_argument('-env', type=str, default="Pong-v4")
parser.add_argument('-agent_file', type=str, default="muzero")
parser.add_argument('-buffer_file', type=str, default="buffer")
parser.add_argument('-buffer_size', type=int, default=3)
parser.add_argument('-td_step', type=int, default=10)
parser.add_argument('-rollout_steps', type=int, default=5)
parser.add_argument('-prev_obs', type=int, default=32)
parser.add_argument('-n_games', type=int, default=10)
parser.add_argument('-n_epoch', type=int, default=250)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-frameskip', type=int, default=2)
parser.add_argument('-rand_act', type=float, default=0.8,
                    help="Percentage of actions done randomly.")
parser.add_argument('-root_noise', type=float, default=0.8,
                    help="Add noise to probabilities in root.")
parser.add_argument('-mcts_simulations', type=int, default=25)
parser.add_argument(
    '-play', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument(
    '-train', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument(
    '-render', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument(
    '-play_train', action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

if args.render:
    env = gym.make("Pong-v4", render_mode="human", frameskip=args.frameskip)
else:
    env = gym.make("Pong-v4", frameskip=args.frameskip)

agent_config = {
    "observation_dim": [3, 96, 96],
    "observation_history": args.prev_obs,
    "rollout_steps": args.rollout_steps,
    "representation_outputs": [4, 4, 2],
    "dynamic_hidden_size": [18, 20, 8],
    "support_size": 21,
    "action_space": env.action_space.n,
    "prediction_hidden_size": [8, 11, 6],
    "mcts_root_exploration": args.root_noise,
    "root_dirichlet_alpha": 0.25,
    "mcts_max_depth": args.rollout_steps * 2,
    "mcts_discount_value": 0.8,
    "mcts_simulations": args.mcts_simulations,
    "pUCT_score_c1": 1.25,
    "pUCT_score_c2": 19652,
    "weight_decay": 1e-4,
    "momentum": 0.9,
    "lr_init": 0.05,
    "lr_decay_rate": 0.1,
    "lr_decay_steps": 350,
    "random_action_threshold": args.rand_act,
    "batch_size": args.batch_size,
}
try:
    with open("assets/" + args.agent_file, "rb") as f:
        agent = pickle.load(f)
    agent.config["random_action_threshold"] = args.rand_act
    agent.config["mcts_simulations"] = args.mcts_simulations

    g = Game(env, agent.config, args.agent_file,
             args.buffer_file, args.buffer_size)
except:
    g = Game(env, agent_config, args.agent_file,
             args.buffer_file, args.buffer_size)
if args.train:
    g.train_agent(args.n_epoch)
if args.play:
    g.run(args.n_games)
if args.play_train:
    for i in range(args.n_games):
        g.play_game()
        g.train_agent(args.n_epoch)
