#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import os
import torch
import argparse
import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as anim
import yaml

from imitate_episodes import make_policy
from utils import _get_episode_paths, load_npz_episode, split_indices, set_seed

import IPython
e = IPython.embed

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str)
parser.add_argument("--mode", type=str, default="train")
parser.add_argument("--idx", type=int, default=0)
parser.add_argument("--output", choices=('image', 'video'), default='image')
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
if mode == 'test':
    mode = 'val'
if mode not in ('train', 'val'):
    raise ValueError(f"mode must be 'train', 'val', or 'test', got {args.mode}")

episode_paths = _get_episode_paths(config['dataset_dir'])
train_indices, val_indices = split_indices(
    len(episode_paths),
    train_ratio=config.get('train_ratio', 0.8),
    shuffle=config.get('shuffle', True),
    seed=config.get('seed', None),
)
selected_indices = train_indices if mode == 'train' else val_indices
if idx < 0 or idx >= len(selected_indices):
    raise IndexError(f'idx {idx} is out of range for {mode} split of size {len(selected_indices)}')

episode_path = episode_paths[selected_indices[idx]]
episode = load_npz_episode(
    episode_path,
    config['camera_names'],
    config.get('camera_height', 480),
    config.get('camera_width', 640),
)

images = np.stack([episode['camera_images'][cam_name] for cam_name in config['camera_names']], axis=0)
robot_states = episode['qpos']
print(images.shape)
episode_len = robot_states.shape[0]
robot_state_dim = robot_states.shape[-1]

set_seed(1)
# command line parameters
policy_class = config['policy_class']

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
        img_t = torch.from_numpy(images[:, loop_ct]).permute(0, 3, 1, 2)
        img_t = (img_t / 255.0).float().to(device).unsqueeze(0)
        joint_t = pre_process(robot_states[loop_ct])
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


def anim_update(i):
    for j in range(camera_num + 1):
        ax[j].cla()

    for camera_id in range(camera_num):
        ax[camera_id].imshow(images[camera_id, i])
        ax[camera_id].axis("off")
        ax[camera_id].set_title("{}".format(config['camera_names'][camera_id]))

    # plot pose
    ax[-1].set_xlim(0, T)
    ax[-1].plot(robot_states[1:], linestyle="dashed", c="k")
    for robot_state_idx in range(robot_state_dim):
        ax[-1].plot(np.arange(i + 1), target_qpos[: i + 1, robot_state_idx])
    ax[-1].set_xlabel("Step")
    ax[-1].set_title("Robot States")


tag = os.path.split((os.path.split(args.ckpt_path)[0]))[1]
epoch = os.path.splitext(os.path.basename(args.ckpt_path))[0]

if args.output == 'video':
    T = len(robot_states)
    camera_num = len(config['camera_names'])
    fig, ax = plt.subplots(1, camera_num + 1, figsize=(12, 5), dpi=60)
    plt.suptitle('temporal ensenble: {}, num_queries: {}'.format(config['temporal_agg'], config['num_queries']))
    ani = anim.FuncAnimation(fig, anim_update, frames=T)
    ani.save("./output/{}_{}_{}_{}.mp4".format(tag, epoch, mode, idx), fps=10, writer="ffmpeg")
else:
    # plot image
    plt.xlim(0,nloop - 1)
    plt.plot(robot_states[1:], linestyle='dashed', c='k')
    plt.plot(target_qpos[:-1])
    plt.title('temporal ensenble: {}, num_queries: {}'.format(config['temporal_agg'], config['num_queries']))
    plt.savefig('./output/{}_{}_{}_{}.png'.format(tag, epoch, mode, idx))