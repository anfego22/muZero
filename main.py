import argparse
import gym
from game.game import Game

ROLLOUT_STEPS = 6
RANDOM_ACTION_DECAY = 0.995


parser = argparse.ArgumentParser()
parser.add_argument('-env', type=str, default="Pong-v4")
parser.add_argument('-agent_file', type=str, default="muzero")
parser.add_argument('-buffer_file', type=str, default="buffer")
parser.add_argument('-buffer_size', type=int, default=3)
parser.add_argument('-td_step', type=int, default=10)
parser.add_argument('-prev_obs', type=int, default=32)
parser.add_argument('-n_games', type=int, default=10)
parser.add_argument('-n_epoch', type=int, default=250)
parser.add_argument('-frameskip', type=int, default=2)
parser.add_argument('-rand_act', type=float, default=0.8,
                    help="Percentage of actions done randomly.")
parser.add_argument('-root_noise', type=float, default=0.8,
                    help="Add noise to probabilities in root.")
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

g = Game(env, args.rand_act, args.agent_file,
         args.buffer_file, args.buffer_size, args.td_step, args.root_noise, args.prev_obs)
if args.train:
    g.train_agent(args.n_epoch)
if args.play:
    g.run(args.n_games)
if args.play_train:
    for i in range(args.n_games):
        g.play_game()
        g.train_agent(args.n_epoch)
