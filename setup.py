#!/usr/bin/env python3
"""One-time dataset setup for TinyWorlds autoresearch.

This script generates an action-labeled low-complexity visual world-modeling
dataset from MiniGrid and stores it as HDF5.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


TIME_BUDGET = 300
DEFAULT_DATA_PATH = Path("data/minigrid_empty8x8_actions.h5")
DEFAULT_ENV_ID = "MiniGrid-Empty-8x8-v0"
DEFAULT_EPISODES = 128
DEFAULT_MAX_STEPS = 128
DEFAULT_FRAME_SIZE = 64
DEFAULT_NUM_ACTIONS = 7

ACTION_NAMES = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an action-labeled MiniGrid HDF5 dataset."
    )
    parser.add_argument(
        "--env-id",
        default=DEFAULT_ENV_ID,
        help="Gymnasium/MiniGrid environment id.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Output HDF5 path.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help="Number of episodes to generate.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum steps per episode.",
    )
    parser.add_argument(
        "--frame-size",
        type=int,
        default=DEFAULT_FRAME_SIZE,
        help="Square RGB frame size in pixels.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed.",
    )
    parser.add_argument(
        "--policy",
        choices=("random", "coverage"),
        default="coverage",
        help="Data collection policy. Coverage biases toward visible movement.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Episode fraction assigned to train split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Episode fraction assigned to validation split.",
    )
    return parser.parse_args()


def require_minigrid():
    try:
        import gymnasium as gym
        import minigrid  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing MiniGrid dependencies. Install them with:\n"
            "  python3 -m pip install 'gymnasium>=1.0.0' 'minigrid>=3.0.0'\n"
        ) from exc
    return gym


def resize_frame(frame: np.ndarray, frame_size: int) -> np.ndarray:
    if frame.shape[0] == frame_size and frame.shape[1] == frame_size:
        return frame.astype(np.uint8)
    return cv2.resize(frame, (frame_size, frame_size), interpolation=cv2.INTER_AREA).astype(
        np.uint8
    )


def choose_action(
    rng: np.random.Generator,
    action_n: int,
    policy: str,
    step: int,
) -> int:
    if policy == "random":
        return int(rng.integers(action_n))

    # A tiny hand-built exploration prior: turn sometimes, move forward often,
    # and occasionally sample every legal action so action IDs remain covered.
    if step % 17 == 0:
        return int(rng.integers(action_n))
    return int(rng.choice([0, 1, 2], p=[0.2, 0.2, 0.6]))


def collect_dataset(args: argparse.Namespace) -> Tuple[Dict[str, np.ndarray], Dict]:
    gym = require_minigrid()
    rng = np.random.default_rng(args.seed)
    env = gym.make(args.env_id, render_mode="rgb_array", max_steps=args.max_steps)

    frames: List[np.ndarray] = []
    actions: List[int] = []
    rewards: List[float] = []
    terminals: List[bool] = []
    episode_starts: List[int] = []
    episode_lengths: List[int] = []

    for episode in range(args.episodes):
        env.reset(seed=args.seed + episode)
        episode_starts.append(len(frames))
        frame = resize_frame(env.render(), args.frame_size)
        frames.append(frame)

        for step in range(args.max_steps):
            action = choose_action(rng, env.action_space.n, args.policy, step)
            _, reward, terminated, truncated, _ = env.step(action)
            next_frame = resize_frame(env.render(), args.frame_size)

            actions.append(action)
            rewards.append(float(reward))
            terminals.append(bool(terminated or truncated))
            frames.append(next_frame)

            if terminated or truncated:
                break

        episode_lengths.append(len(frames) - episode_starts[-1])

    env.close()

    arrays = {
        "frames": np.asarray(frames, dtype=np.uint8),
        "actions": np.asarray(actions, dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "terminals": np.asarray(terminals, dtype=np.bool_),
        "episode_starts": np.asarray(episode_starts, dtype=np.int64),
        "episode_lengths": np.asarray(episode_lengths, dtype=np.int64),
    }
    metadata = {
        "env_id": args.env_id,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "frame_size": args.frame_size,
        "seed": args.seed,
        "policy": args.policy,
        "action_names": {
            str(i): ACTION_NAMES.get(i, f"action_{i}") for i in range(env.action_space.n)
        },
        "schema": {
            "frames": "[N, H, W, C] uint8",
            "actions": "[N - episodes] int64; action[t] maps frame[t] to frame[t + 1] within an episode",
            "rewards": "per action",
            "terminals": "per action",
        },
    }
    return arrays, metadata


def episode_split_ranges(
    episode_starts: np.ndarray,
    episode_lengths: np.ndarray,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, np.ndarray]:
    n_episodes = len(episode_starts)
    train_end = int(n_episodes * train_ratio)
    val_end = int(n_episodes * (train_ratio + val_ratio))
    split_episode_ranges = {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, n_episodes),
    }

    split_frame_ranges: Dict[str, np.ndarray] = {}
    for name, (start_ep, end_ep) in split_episode_ranges.items():
        if start_ep >= end_ep:
            split_frame_ranges[name] = np.asarray([0, 0], dtype=np.int64)
            continue
        start_frame = int(episode_starts[start_ep])
        last_ep = end_ep - 1
        end_frame = int(episode_starts[last_ep] + episode_lengths[last_ep])
        split_frame_ranges[name] = np.asarray([start_frame, end_frame], dtype=np.int64)
    return split_frame_ranges


def write_h5(
    out: Path,
    arrays: Dict[str, np.ndarray],
    metadata: Dict,
    train_ratio: float,
    val_ratio: float,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    split_ranges = episode_split_ranges(
        arrays["episode_starts"], arrays["episode_lengths"], train_ratio, val_ratio
    )

    with h5py.File(out, "w") as h5:
        h5.create_dataset("frames", data=arrays["frames"], compression="lzf")
        h5.create_dataset("actions", data=arrays["actions"], compression="lzf")
        h5.create_dataset("rewards", data=arrays["rewards"], compression="lzf")
        h5.create_dataset("terminals", data=arrays["terminals"], compression="lzf")
        h5.create_dataset("episode_starts", data=arrays["episode_starts"])
        h5.create_dataset("episode_lengths", data=arrays["episode_lengths"])
        splits = h5.create_group("splits")
        for name, frame_range in split_ranges.items():
            splits.create_dataset(name, data=frame_range)
        h5.attrs["metadata"] = json.dumps(metadata, sort_keys=True)


class MiniGridTransitionDataset(Dataset):
    def __init__(
        self,
        path: Union[Path, str] = DEFAULT_DATA_PATH,
        split: str = "train",
        context_length: int = 2,
    ) -> None:
        self.path = Path(path)
        self.split = split
        self.context_length = context_length

        with h5py.File(self.path, "r") as h5:
            self.frames = h5["frames"][:]
            self.actions = h5["actions"][:]
            self.episode_starts = h5["episode_starts"][:]
            self.episode_lengths = h5["episode_lengths"][:]
            metadata_raw = h5.attrs.get("metadata", "{}")
            self.metadata = json.loads(metadata_raw)
            if "splits" in h5 and split in h5["splits"]:
                self.frame_range = tuple(int(v) for v in h5["splits"][split][:])
            else:
                self.frame_range = (0, len(self.frames))

        self.num_actions = len(self.metadata.get("action_names", ACTION_NAMES))
        self.indices = self._build_indices()

    def _build_indices(self) -> List[Tuple[int, int]]:
        valid: List[Tuple[int, int]] = []
        split_start, split_end = self.frame_range
        for episode_idx, (start, length) in enumerate(
            zip(self.episode_starts, self.episode_lengths)
        ):
            end = int(start + length)
            if end <= split_start or int(start) >= split_end:
                continue
            first_i = max(int(start) + self.context_length - 1, split_start)
            last_i = min(end - 2, split_end - 2)
            for frame_idx in range(first_i, last_i + 1):
                # Actions omit the first frame of each episode.
                action_idx = frame_idx - episode_idx
                valid.append((frame_idx, action_idx))
        return valid

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        frame_idx, action_idx = self.indices[index]
        context = self.frames[
            frame_idx - self.context_length + 1 : frame_idx + 1
        ].astype(np.float32) / 255.0
        target = self.frames[frame_idx + 1].astype(np.float32) / 255.0
        context = torch.from_numpy(context).permute(0, 3, 1, 2).contiguous()
        target = torch.from_numpy(target).permute(2, 0, 1).contiguous()
        action = torch.tensor(int(self.actions[action_idx]), dtype=torch.long)
        return context, action, target


def make_dataloader(
    data_path: Union[Path, str],
    split: str,
    context_length: int,
    batch_size: int,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
) -> DataLoader:
    dataset = MiniGridTransitionDataset(data_path, split, context_length)
    if len(dataset) == 0:
        raise ValueError(
            f"No transitions found for split={split!r}, context_length={context_length}"
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train") if shuffle is None else shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
    )


def ensure_dataset(
    path: Union[Path, str] = DEFAULT_DATA_PATH,
    episodes: int = DEFAULT_EPISODES,
    max_steps: int = DEFAULT_MAX_STEPS,
    frame_size: int = DEFAULT_FRAME_SIZE,
    env_id: str = DEFAULT_ENV_ID,
) -> Path:
    path = Path(path)
    if path.exists():
        return path
    args = argparse.Namespace(
        env_id=env_id,
        out=path,
        episodes=episodes,
        max_steps=max_steps,
        frame_size=frame_size,
        seed=17,
        policy="coverage",
        train_ratio=0.8,
        val_ratio=0.1,
    )
    arrays, metadata = collect_dataset(args)
    write_h5(path, arrays, metadata, args.train_ratio, args.val_ratio)
    return path


@torch.no_grad()
def evaluate_world_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: Union[torch.device, str],
    num_actions: int = DEFAULT_NUM_ACTIONS,
    max_batches: int = 16,
) -> Dict[str, float]:
    model.eval()
    device = torch.device(device)

    logged_mse_sum = 0.0
    margin_sum = 0.0
    spread_sum = 0.0
    n_examples = 0

    for batch_idx, (context, action, target) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        context = context.to(device, non_blocking=True)
        action = action.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        bsz = context.shape[0]

        predict_next = getattr(model, "predict_next", model)
        pred_true = predict_next(context, action).clamp(0.0, 1.0)
        mse_true_per = F.mse_loss(pred_true, target, reduction="none").flatten(1).mean(1)

        per_action_preds = []
        per_action_mse = []
        for action_id in range(num_actions):
            cf_action = torch.full_like(action, action_id)
            pred = predict_next(context, cf_action).clamp(0.0, 1.0)
            per_action_preds.append(pred)
            per_action_mse.append(
                F.mse_loss(pred, target, reduction="none").flatten(1).mean(1)
            )

        all_preds = torch.stack(per_action_preds, dim=1)
        all_mse = torch.stack(per_action_mse, dim=1)
        action_mask = F.one_hot(action, num_classes=num_actions).bool()
        wrong_mse = all_mse.masked_fill(action_mask, float("inf")).min(dim=1).values
        margin = wrong_mse - mse_true_per

        if num_actions > 1:
            diffs = all_preds[:, :, None] - all_preds[:, None, :]
            pairwise = diffs.square().flatten(3).mean(3)
            upper = torch.triu(torch.ones(num_actions, num_actions, device=device), diagonal=1).bool()
            spread_per = pairwise[:, upper].mean(1)
        else:
            spread_per = torch.zeros_like(mse_true_per)

        logged_mse_sum += float(mse_true_per.sum().item())
        margin_sum += float(margin.sum().item())
        spread_sum += float(spread_per.sum().item())
        n_examples += bsz

    if n_examples == 0:
        raise ValueError("Evaluation loader produced no examples.")

    logged_mse = logged_mse_sum / n_examples
    action_margin = margin_sum / n_examples
    counterfactual_spread = spread_sum / n_examples
    score = logged_mse - 0.1 * max(action_margin, 0.0)
    return {
        "val_mse": logged_mse,
        "action_margin": action_margin,
        "counterfactual_spread": counterfactual_spread,
        "score": score,
    }


def main() -> int:
    args = parse_args()
    arrays, metadata = collect_dataset(args)
    write_h5(args.out, arrays, metadata, args.train_ratio, args.val_ratio)

    print(f"Wrote {args.out}")
    print(f"frames: {arrays['frames'].shape}")
    print(f"actions: {arrays['actions'].shape}")
    print(f"episodes: {len(arrays['episode_starts'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
