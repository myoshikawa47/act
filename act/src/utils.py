import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import IPython

e = IPython.embed

LEFT_QPOS_KEY = 'left_arm_command'
RIGHT_QPOS_KEY = 'right_arm_command'


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_paths, camera_names, norm_stats, camera_height, camera_width, max_episode_len):
        super(EpisodicDataset).__init__()
        self.episode_paths = [Path(path) for path in episode_paths]
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.max_episode_len = max_episode_len
        self.is_sim = False
        self.episodes = [
            load_npz_episode(path, camera_names, camera_height, camera_width)
            for path in self.episode_paths
        ]

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        sample_full_episode = False

        episode = self.episodes[index]
        joints = episode['qpos']
        episode_len = episode['episode_len']

        if sample_full_episode:
            start_ts = 0
        else:
            start_ts = np.random.choice(episode_len)

        qpos = joints[start_ts]
        action = joints[start_ts + 1:episode_len]
        action_len = episode_len - (start_ts + 1)

        padded_action = np.zeros((self.max_episode_len, joints.shape[1]), dtype=np.float32)
        if action_len > 0:
            padded_action[:action_len] = action
        is_pad = np.ones(self.max_episode_len, dtype=bool)
        is_pad[:action_len] = False

        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(episode['camera_images'][cam_name][start_ts])
        all_cam_images = np.stack(all_cam_images, axis=0)

        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad)

        image_data = torch.einsum('k h w c -> k c h w', image_data)
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats['action_mean']) / self.norm_stats['action_std']
        qpos_data = (qpos_data - self.norm_stats['qpos_mean']) / self.norm_stats['qpos_std']

        return image_data, qpos_data, action_data, is_pad


def _decode_image(encoded_frame, episode_path, cam_name, frame_idx, camera_height, camera_width):
    if isinstance(encoded_frame, np.ndarray):
        encoded_frame = encoded_frame.astype(np.uint8, copy=False)
    else:
        encoded_frame = np.frombuffer(encoded_frame, dtype=np.uint8)

    image = cv2.imdecode(encoded_frame, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            f"Failed to decode compressed image for {episode_path}:{cam_name}[{frame_idx}]"
        )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (camera_width, camera_height), interpolation=cv2.INTER_LINEAR)


def _decode_image_sequence(encoded_frames, episode_path, cam_name, camera_height, camera_width):
    decoded_frames = []
    for frame_idx, encoded_frame in enumerate(encoded_frames):
        decoded_frames.append(
            _decode_image(
                encoded_frame,
                episode_path,
                cam_name,
                frame_idx,
                camera_height,
                camera_width,
            )
        )
    return np.stack(decoded_frames, axis=0)


def _get_qpos_arrays(data):
    left_joint = data[LEFT_QPOS_KEY].astype(np.float32)
    right_joint = data[RIGHT_QPOS_KEY].astype(np.float32)
    return left_joint, right_joint


def _get_episode_len_from_arrays(arrays, camera_names):
    left_joint, right_joint = _get_qpos_arrays(arrays)
    lengths = [left_joint.shape[0], right_joint.shape[0]]
    lengths.extend(arrays[cam_name].shape[0] for cam_name in camera_names)
    episode_len = min(lengths)
    if episode_len <= 0:
        raise ValueError('Episode has no valid timesteps.')
    return episode_len


def _build_qpos_from_arrays(arrays, episode_len):
    left_joint, right_joint = _get_qpos_arrays(arrays)
    return np.concatenate(
        [left_joint[:episode_len], right_joint[:episode_len]],
        axis=-1,
    )


def load_npz_episode(npz_path, camera_names, camera_height, camera_width):
    episode_path = Path(npz_path)
    with np.load(episode_path, allow_pickle=True) as data:
        arrays = {key: data[key] for key in data.files}

    missing_cameras = [cam_name for cam_name in camera_names if cam_name not in arrays]
    if missing_cameras:
        raise KeyError(f"Missing camera keys {missing_cameras} in {episode_path}")

    episode_len = _get_episode_len_from_arrays(arrays, camera_names)
    qpos = _build_qpos_from_arrays(arrays, episode_len)
    camera_images = {
        cam_name: _decode_image_sequence(
            arrays[cam_name][:episode_len],
            episode_path,
            cam_name,
            camera_height,
            camera_width,
        )
        for cam_name in camera_names
    }

    return {
        'path': episode_path,
        'qpos': qpos,
        'camera_images': camera_images,
        'episode_len': episode_len,
    }


def _get_episode_paths(dataset_dir):
    episode_paths = sorted(Path(dataset_dir).glob('*.npz'))
    if not episode_paths:
        raise FileNotFoundError(f'No .npz files found in {dataset_dir}')
    return episode_paths


def _get_episode_len_from_npz(npz_path, camera_names):
    with np.load(npz_path, allow_pickle=True) as data:
        arrays = {key: data[key] for key in data.files}
    return _get_episode_len_from_arrays(arrays, camera_names)


def _build_qpos_from_npz(npz_path, camera_names):
    with np.load(npz_path, allow_pickle=True) as data:
        arrays = {key: data[key] for key in data.files}

    episode_len = _get_episode_len_from_arrays(arrays, camera_names)
    return _build_qpos_from_arrays(arrays, episode_len)


def get_norm_stats(dataset_dir, camera_names):
    all_qpos_data = []
    episode_paths = _get_episode_paths(dataset_dir)
    for npz_path in episode_paths:
        qpos = _build_qpos_from_npz(npz_path, camera_names)
        all_qpos_data.append(torch.from_numpy(qpos))

    all_qpos_data = torch.cat(all_qpos_data, dim=0)

    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
    qpos_std = all_qpos_data.std(dim=0, keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)

    action_mean, action_std = qpos_mean, qpos_std

    stats = {
        'action_mean': action_mean.numpy().squeeze(),
        'action_std': action_std.numpy().squeeze(),
        'qpos_mean': qpos_mean.numpy().squeeze(),
        'qpos_std': qpos_std.numpy().squeeze(),
        'example_qpos': qpos,
    }

    return stats


def split_indices(num_episodes, train_ratio=0.8, shuffle=True, seed=None):
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f'train_ratio must satisfy 0.0 < train_ratio < 1.0, got {train_ratio}')

    indices = np.arange(num_episodes)
    if shuffle:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(indices)

    split_index = int(num_episodes * train_ratio)
    train_indices = indices[:split_index]
    val_indices = indices[split_index:]
    return train_indices, val_indices


def load_data(dataset_dir, camera_names, batch_size_train, batch_size_val, train_ratio=0.8, shuffle=True, seed=None, camera_height=480, camera_width=640):
    print(f'\nData from: {dataset_dir}\n')

    episode_paths = _get_episode_paths(dataset_dir)
    train_indices, val_indices = split_indices(len(episode_paths), train_ratio, shuffle, seed)

    train_episode_paths = [episode_paths[idx] for idx in train_indices]
    val_episode_paths = [episode_paths[idx] for idx in val_indices]

    norm_stats = get_norm_stats(dataset_dir, camera_names)
    max_episode_len = max(_get_episode_len_from_npz(path, camera_names) for path in episode_paths)

    num_workers = min(8, max(1, (os.cpu_count() or 1) // 2))

    train_dataset = EpisodicDataset(
        train_episode_paths,
        camera_names,
        norm_stats,
        camera_height,
        camera_width,
        max_episode_len,
    )
    val_dataset = EpisodicDataset(
        val_episode_paths,
        camera_names,
        norm_stats,
        camera_height,
        camera_width,
        max_episode_len,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=len(train_dataset) > 0,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=len(val_dataset) > 0,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
    )

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
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

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
