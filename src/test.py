#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import os
import glob
import sys
import torch
import argparse
import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as anim
import yaml
import time

import pickle
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from imitate_episodes import make_policy
from utils import compute_dict_mean, set_seed, detach_dict # helper functions

import IPython
e = IPython.embed

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str)
parser.add_argument("--mode", type=str, default="test")
parser.add_argument("--idx", type=int, default=1)
parser.add_argument("--device", type=int, default=0)

args = parser.parse_args()

if args.device >= 0:
        device = 'cuda:{}'.format(args.device)
else:
    device = 'cpu'

with open(os.path.join(os.path.split(args.ckpt_path)[0], 'config.yaml')) as f:
    config = yaml.safe_load(f)
config['device'] = device

mode = args.mode
idx = args.idx

# load dataset
images = []
for cam_name in config['camera_names']:
    images.append(np.load('{}/{}/{}.npy'.format(config['dataset_dir'], mode, cam_name))[idx])
images = np.array(images)
print(images.shape)
episode_len = images.shape[1]
robot_states = np.load('{}/{}/robot_states.npy'.format(config['dataset_dir'], mode))[idx]
robot_state_dim = robot_states.shape[-1] # 4


# define model
set_seed(1)
# command line parameters
policy_class = config['policy_class']

policy = make_policy(policy_class, config)
policy = make_policy(policy_class, config)
policy.load_state_dict(torch.load(args.ckpt_path, map_location=torch.device(device)))
policy.eval()

# load stats
stats_path = os.path.join(os.path.split(args.ckpt_path)[0], f'dataset_stats.npz')
stats = np.load(stats_path)

pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
post_process = lambda a: a * stats['action_std'] + stats['action_mean']

query_frequency = config['num_queries']
if config['temporal_agg']:
    query_frequency = 1
    num_queries = config['num_queries']


# Inference
if config['temporal_agg']:
    all_time_actions = torch.zeros([episode_len, episode_len+num_queries, robot_state_dim]).to(device)
target_qpos_list = []
nloop = episode_len
with torch.inference_mode():
    for loop_ct in range(nloop):
        # load data and normalization
        img_t = images[:,loop_ct].transpose(0, 3, 1, 2)
        img_t = torch.from_numpy(img_t / 255.0).float().to(device).unsqueeze(0)
        joint_t = robot_states[loop_ct]
        joint_t = pre_process(joint_t)
        joint_t = torch.from_numpy(joint_t).float().to(device).unsqueeze(0)

        # prediction
        if loop_ct % query_frequency == 0:
            all_actions = policy(joint_t, img_t)
        if config['temporal_agg']:
            all_time_actions[loop_ct, loop_ct:loop_ct+num_queries] = all_actions
            actions_for_curr_step = all_time_actions[:, loop_ct]
            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]
            k = 0.01
            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = torch.from_numpy(exp_weights).to(device).unsqueeze(dim=1)
            raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
        else:
            raw_action = all_actions[:, loop_ct % query_frequency]

        # denormalization
        raw_action = raw_action.squeeze(0).cpu().detach().numpy()
        action = post_process(raw_action)
        target_qpos = action

        # append data
        target_qpos_list.append(target_qpos)
        
        print("loop_ct:{}, target_qpos:{}".format(loop_ct, target_qpos))

target_qpos = np.array(target_qpos_list)

# # plot images
# T = len(images)
# fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=60)


# def anim_update(i):
#     for j in range(2):
#         ax[j].cla()

#     # plot camera image
#     ax[0].imshow(images[i, :, :, ::-1])
#     ax[0].axis("off")
#     ax[0].set_title("Input image")

#     # plot pose
#     # ax[2].set_ylim(-np.pi / 2, np.pi / 2)
#     # ax[2].set_ylim(-2.0, 2.0)
#     ax[1].set_xlim(0, T)
#     ax[1].plot(robot_states[1:], linestyle="dashed", c="k")
#     for robot_state_idx in range(robot_state_dim):
#         ax[1].plot(np.arange(i + 1), target_qpos[: i + 1, robot_state_idx])
#     ax[1].set_xlabel("Step")
#     ax[1].set_title("Robot States")


# ani = anim.FuncAnimation(fig, anim_update, frames=T)
# ani.save("/home/yoshikawa/job/2024/airec/ACT/output/{}_{}_{}.mp4".format(os.path.split(args['ckpt_path'])[-1], args['mode'], idx), fps=10, writer="ffmpeg")

# plot image
plt.xlim(0,nloop - 1)
plt.plot(robot_states[1:], linestyle='dashed', c='k')
plt.plot(target_qpos[:-1])
plt.title('TA: {}, num_queries: {}'.format(config['temporal_agg'], config['num_queries']))
tag = os.path.split((os.path.split(args.ckpt_path)[0]))[1]
epoch = os.path.splitext(os.path.basename(args.ckpt_path))[0]
plt.savefig('./output/{}_{}_{}_{}.png'.format(tag, epoch, mode, idx))