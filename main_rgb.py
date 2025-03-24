
import argparse
import numpy as np
import itertools
import torch
from drqv2 import *
from replay_memory import ReplayMemory
from tqdm import tqdm
import cv2
from environment2.envs import ENV
from model import *
import pandas as pd

parser = argparse.ArgumentParser(description='PyTorch Drq-v2 Args')
parser.add_argument('--env-name', default="inverted_pendulum_series",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=False,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.01, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.1, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--seed', type=int, default=123458, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=1024, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=5000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=2, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=150000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--agent_name', type=str, default='DrQV2',
                    help='run on CUDA (default: False)')
args = parser.parse_args()

task_name = "CoCELSIP"
render_mode = "off_screen"
lookat_default = np.array([0, -0.1, 0.4])
distance_default = 2.4
elevation_default = 0
azimuth_default = 180
evaludation_period = 10

env = ENV(task_name, lookat=lookat_default, distance=distance_default, elevation=elevation_default, azimuth=azimuth_default, n_history=3, render_mode=render_mode)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
total_numsteps = 0
updates = 0
device = torch.device("cuda")

# Hyper-parameterc
stack = 3
scale = 6
channel = 1
img_size = 80
stddev_schedule = 'linear(1.0,0.1,100000)'
encoder_layers = 3
encoder_filters = 32
feature_dim = 50

agent = DrQV2(env.action_space, args, stddev_schedule, img_size=img_size, feature_dim = feature_dim, encoder_layers=encoder_layers, encoder_filters=encoder_filters, frame_stack = channel*stack)

if args.eval is True:
    agent.load_checkpoint("checkpoints/drqv2_checkpoint_inverted_pendulum_series_", evaluate=True)

memory = ReplayMemory(args.replay_size, args.seed)
monitoring_file = './runs/pendulum.csv'
os.makedirs('./runs/', exist_ok=True)
file = pd.DataFrame(columns=["Episode", "Total Steps", "Episode Steps", "Total Rwd"])
file.to_csv(monitoring_file, index=False)


for i_episode in itertools.count(1):
    episode_reward = 0
    episode_rand_reward = 0
    episode_steps = 0
    done = False
    env.reset()
    state_array = env.render(azimuth_default, distance_default, elevation_default, azimuth_default) 

    state_array = state_array[:, 80:-80, :]
    h, w, _ = state_array.shape
    state_array = cv2.cvtColor(state_array,cv2.COLOR_BGR2GRAY)
    state_array = cv2.resize(state_array, (int(w/scale), int(h/scale)))

    state = state_array.reshape(channel, int(w/scale), int(h/scale)) 
    state = torch.FloatTensor(state.copy()).to(device)
    state = torch.cat([state, state, state], 0) 

    for count in tqdm(range(args.num_steps)):
        if args.start_steps > total_numsteps and args.eval is False: 
            action = env.action_space.sample() 
        elif args.eval is True:
            action = agent.select_action(state, total_numsteps, evaluate=True) 
        else:
            action = agent.select_action(state, total_numsteps)

        if len(memory) > args.batch_size and args.eval is False:
            for i in range(args.updates_per_step):
                critic_1_loss, critic_2_loss, policy_loss = agent.update_parameters_gpu(memory, args.batch_size, updates, total_numsteps)
                updates += 1

        _, reward, done, _, _ = env.step(action * env.action_max)

        next_state_array = env.render(lookat_default, distance_default, elevation_default, azimuth_default)
        next_state_array = next_state_array[:, 80:-80, :]

        next_state_array = cv2.cvtColor(next_state_array,cv2.COLOR_BGR2GRAY)
        next_state_array = cv2.resize(next_state_array, (int(w/scale), int(h/scale)))

        next_state = next_state_array.reshape(channel, int(w/scale), int(h/scale))
        next_state = torch.FloatTensor(next_state.copy()).to(device)
        next_state = torch.cat([state[channel:], next_state], 0) 

        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        
        if args.eval is False:
            memory.push(state, torch.FloatTensor(action).to('cuda'), torch.FloatTensor(np.array([reward])).to('cuda'), next_state, torch.FloatTensor(np.array([mask])).to('cuda'))

        cv2.imshow('Rendering', env.render(lookat_default, distance_default, elevation_default, azimuth_default))
        cv2.imshow('state', next_state_array)
        cv2.waitKey(1)
        state_array = next_state_array
        state = next_state  
        
        if done: break
        
    print("Train Episode: {}, total numsteps: {}, episode steps: {}, average reward: {}".format(i_episode, total_numsteps, episode_steps, np.round(episode_reward, 2)))
    file = pd.read_csv(monitoring_file)
    file.loc[len(file)] = [i_episode, total_numsteps, episode_steps, np.round(episode_reward, 2)]
    file.to_csv(monitoring_file, index=False)

    agent.save_checkpoint(args.env_name)

