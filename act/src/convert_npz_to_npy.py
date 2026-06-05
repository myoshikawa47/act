import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert episodic NPZ files into padded NPY tensors."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing source .npz files. Each .npz is treated as one episode.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where train/ and test/ subdirectories will be created.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of episodes assigned to the train split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when shuffling episodes before splitting.",
    )
    return parser.parse_args()


def load_episodes(npz_paths):
    episodes = []
    expected_keys = None

    for npz_path in npz_paths:
        with np.load(npz_path, allow_pickle=True) as data:
            keys = sorted(data.files)
            if expected_keys is None:
                expected_keys = keys
            elif keys != expected_keys:
                raise ValueError(
                    f"Inconsistent keys in {npz_path}. "
                    f"Expected {expected_keys}, got {keys}."
                )

            episode = {}
            for key in expected_keys:
                array = maybe_decode_image_sequence(data[key], npz_path, key)
                if array.ndim == 0:
                    raise ValueError(
                        f"{npz_path}:{key} is scalar. Expected at least 1 dimension "
                        "with the first axis as sequence length."
                    )
                if array.shape[0] == 0:
                    raise ValueError(
                        f"{npz_path}:{key} has zero sequence length and cannot be padded "
                        "with the terminal value."
                    )
                episode[key] = array
            episodes.append(episode)

    return episodes, expected_keys or []


def split_episodes(episodes, train_ratio, seed):
    if not 0.0 <= train_ratio <= 1.0:
        raise ValueError(f"train_ratio must be between 0.0 and 1.0, got {train_ratio}.")

    if not episodes:
        return [], []

    indices = np.arange(len(episodes))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    train_count = int(len(indices) * train_ratio)
    train_indices = indices[:train_count]
    test_indices = indices[train_count:]

    train_episodes = [episodes[i] for i in train_indices]
    test_episodes = [episodes[i] for i in test_indices]
    return train_episodes, test_episodes


def decode_image_bytes(frame_bytes, npz_path, key, frame_index):
    decoded = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError(
            f"Failed to decode image bytes: {npz_path}:{key}[{frame_index}]"
        )
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def maybe_decode_image_sequence(array, npz_path, key):
    if array.dtype != np.uint8 or array.ndim == 0:
        return array

    if array.ndim == 1:
        decoded = decode_image_bytes(array, npz_path, key, 0)
        return decoded[np.newaxis, ...]

    decoded_frames = []
    reference_shape = None
    for frame_index, frame_bytes in enumerate(array):
        decoded = decode_image_bytes(frame_bytes, npz_path, key, frame_index)
        if reference_shape is None:
            reference_shape = decoded.shape
        elif decoded.shape != reference_shape:
            raise ValueError(
                f"Inconsistent decoded image shape for key '{key}' in {npz_path}. "
                f"Expected {reference_shape}, got {decoded.shape} at frame {frame_index}."
            )
        decoded_frames.append(decoded)

    return np.stack(decoded_frames, axis=0)


def pad_episode_array(array, target_length):
    pad_length = target_length - array.shape[0]
    if pad_length == 0:
        return array

    last_frame = array[-1:]
    pad_block = np.repeat(last_frame, pad_length, axis=0)
    return np.concatenate([array, pad_block], axis=0)


def build_split_arrays(episodes, keys):
    if not episodes:
        return {}

    split_arrays = {}
    for key in keys:
        reference_shape = episodes[0][key].shape[1:]
        reference_dtype = episodes[0][key].dtype

        for episode in episodes[1:]:
            array = episode[key]
            if array.shape[1:] != reference_shape:
                raise ValueError(
                    f"Inconsistent feature shape for key '{key}'. "
                    f"Expected {reference_shape}, got {array.shape[1:]}."
                )
            if array.dtype != reference_dtype:
                raise ValueError(
                    f"Inconsistent dtype for key '{key}'. "
                    f"Expected {reference_dtype}, got {array.dtype}."
                )

        max_length = max(episode[key].shape[0] for episode in episodes)
        padded = [pad_episode_array(episode[key], max_length) for episode in episodes]
        split_arrays[key] = np.stack(padded, axis=0)

    return split_arrays


def save_split(output_dir, split_name, split_arrays):
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    for key, array in split_arrays.items():
        np.save(split_dir / f"{key}.npy", array)
        print(f"Saved {split_dir / f'{key}.npy'}: {array.shape}")


def main():
    args = parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    npz_paths = sorted(input_dir.glob("*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No .npz files found in: {input_dir}")

    episodes, keys = load_episodes(npz_paths)
    train_episodes, test_episodes = split_episodes(episodes, args.train_ratio, args.seed)

    train_arrays = build_split_arrays(train_episodes, keys)
    test_arrays = build_split_arrays(test_episodes, keys)

    if train_episodes:
        save_split(output_dir, "train", train_arrays)
    else:
        print("Skipped train split because it is empty.")

    if test_episodes:
        save_split(output_dir, "test", test_arrays)
    else:
        print("Skipped test split because it is empty.")

    print(
        "Completed conversion: "
        f"{len(train_episodes)} train episodes, {len(test_episodes)} test episodes."
    )


if __name__ == "__main__":
    main()
