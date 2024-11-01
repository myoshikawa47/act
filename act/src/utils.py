import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed

'''
Load data from npy file
'''

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, camera_names, norm_stats, mode):
        super(EpisodicDataset).__init__()
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.mode = mode
        self.is_sim = False
        
        dataset_path = os.path.join(dataset_dir, self.mode)
        self.images = dict()
        for cam_name in self.camera_names:
            self.images[cam_name] = np.load(os.path.join(dataset_path, f'{cam_name}.npy'))
        self.joints = np.load(os.path.join(dataset_path, 'robot_states.npy')) # TODO
        self.len = self.joints.shape[0]
        
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        joints = self.joints[index]
        original_action_shape = joints.shape
        episode_len = original_action_shape[0]

        if sample_full_episode:
            start_ts = 0
        else:
            start_ts = np.random.choice(episode_len)
        # get observation at start_ts only
        qpos = joints[start_ts]
        image_dict = dict()
        for cam_name in self.camera_names:
                image_dict[cam_name] = self.images[cam_name][index][start_ts]
        # get all actions after and including start_ts
        if self.is_sim:
            # action = joints[start_ts:]
            action = joints[start_ts + 1:]
            action_len = episode_len - start_ts
        else:
            # action = joints[max(0, start_ts - 1):] # hack, to make timesteps more aligned
            # action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned
            action = joints[start_ts + 1:] # hack, to make timesteps more aligned
            action_len = episode_len - (start_ts + 1)

        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir):
    all_qpos_data = []
    # all_action_data = []
    for mode in ['train', 'test']:
        qpos = np.load(os.path.join(dataset_dir, mode, 'robot_states.npy')) # TODO
        all_qpos_data.append(torch.from_numpy(qpos))
    all_qpos_data = torch.cat((all_qpos_data), dim=0)

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    # action data is just the same as qpos data
    action_mean, action_std = qpos_mean, qpos_std

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    
    norm_stats = get_norm_stats(dataset_dir)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(dataset_dir, camera_names, norm_stats, mode='train')
    val_dataset = EpisodicDataset(dataset_dir, camera_names, norm_stats, mode='test')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
