import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import IterableDataset
from torchvision.transforms import ColorJitter, InterpolationMode, RandomResizedCrop


@dataclass(frozen=True)
class LiberoTrajectory:
    file_path: str
    demo_key: str
    instruction: str
    num_steps: int


@dataclass(frozen=True)
class LiberoTransition:
    trajectory_idx: int
    step_idx: int


class LiberoMultiviewDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        benchmark_name: str,
        batch_transform,
        camera_view: str = "agentview_rgb",
        train: bool = True,
        image_aug: bool = False,
        shuffle_buffer_size: int = 100_000,
        seed: Optional[int] = None,
        repeat: bool = True,
    ) -> None:
        self.data_root_dir = Path(data_root_dir)
        self.benchmark_name = benchmark_name
        self.batch_transform = batch_transform
        self.camera_view = camera_view
        self.train = train
        self.image_aug = image_aug
        self.shuffle_buffer_size = max(int(shuffle_buffer_size), 1)
        self.seed = seed
        self.repeat = repeat

        self.dataset_dir = self._resolve_dataset_dir(self.data_root_dir, self.benchmark_name)
        self.trajectories, self.transitions, stats = self._index_dataset(self.dataset_dir)
        self._action_q01 = np.asarray(stats["q01"], dtype=np.float32)
        self._action_q99 = np.asarray(stats["q99"], dtype=np.float32)
        self._action_mask = np.asarray(stats["mask"], dtype=bool)
        self._action_min = np.asarray(stats["min"], dtype=np.float32)
        self._action_max = np.asarray(stats["max"], dtype=np.float32)
        self._zeros_mask = self._action_min == self._action_max
        self._num_transitions = int(stats["num_transitions"])
        self._num_trajectories = int(stats["num_trajectories"])

        self.dataset_statistics = {
            self.benchmark_name: {
                "action": {
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "min": stats["min"],
                    "max": stats["max"],
                    "q01": stats["q01"],
                    "q99": stats["q99"],
                    "mask": stats["mask"],
                },
                "proprio": {
                    "mean": stats["proprio_mean"],
                    "std": stats["proprio_std"],
                    "min": stats["proprio_min"],
                    "max": stats["proprio_max"],
                    "q01": stats["proprio_q01"],
                    "q99": stats["proprio_q99"],
                },
                "num_transitions": self._num_transitions,
                "num_trajectories": self._num_trajectories,
            }
        }

    @staticmethod
    def _resolve_dataset_dir(data_root_dir: Path, benchmark_name: str) -> Path:
        candidate = data_root_dir / benchmark_name
        if candidate.is_dir():
            return candidate

        if data_root_dir.is_dir() and any(data_root_dir.glob("*.hdf5")):
            return data_root_dir

        raise FileNotFoundError(f"Could not find LIBERO benchmark directory for `{benchmark_name}` under {data_root_dir}")

    def _index_dataset(
        self, dataset_dir: Path
    ) -> tuple[List[LiberoTrajectory], List[LiberoTransition], Dict[str, Any]]:
        trajectories: List[LiberoTrajectory] = []
        transitions: List[LiberoTransition] = []
        all_actions: List[np.ndarray] = []

        hdf5_files = sorted(dataset_dir.glob("*.hdf5"))
        if not hdf5_files:
            raise FileNotFoundError(f"No .hdf5 files found in {dataset_dir}")

        for hdf5_path in hdf5_files:
            with h5py.File(hdf5_path, "r") as handle:
                data_group = handle["data"]
                problem_info = json.loads(data_group.attrs["problem_info"])
                instruction = str(problem_info["language_instruction"]).strip().lower()

                for demo_key in sorted(data_group.keys()):
                    demo = data_group[demo_key]
                    obs_group = demo["obs"]
                    if self.camera_view not in obs_group:
                        raise KeyError(f"Camera view `{self.camera_view}` not found in {hdf5_path}:{demo_key}/obs")

                    raw_actions = np.asarray(demo["actions"], dtype=np.float32)
                    step_indices = self._get_kept_step_indices(raw_actions)
                    if not step_indices:
                        continue

                    actions = self._standardize_actions(raw_actions)
                    trajectory_idx = len(trajectories)
                    trajectories.append(
                        LiberoTrajectory(
                            file_path=str(hdf5_path),
                            demo_key=demo_key,
                            instruction=instruction,
                            num_steps=len(step_indices),
                        )
                    )
                    transitions.extend(
                        LiberoTransition(trajectory_idx=trajectory_idx, step_idx=int(step_idx))
                        for step_idx in step_indices
                    )
                    all_actions.append(actions[list(step_indices)])

        if not all_actions:
            raise ValueError(f"No valid transitions found in {dataset_dir} after no-op filtering")

        stacked_actions = np.concatenate(all_actions, axis=0)
        q01 = np.quantile(stacked_actions, 0.01, axis=0).astype(np.float32)
        q99 = np.quantile(stacked_actions, 0.99, axis=0).astype(np.float32)
        mask = (q99 > q01).astype(bool)
        # Match the official RLDS LIBERO transform: leave the standardized gripper action
        # in absolute 0/1 space instead of q01/q99-normalizing it.
        mask[-1] = False
        proprio_dim = stacked_actions.shape[1]
        zero_proprio = np.zeros((proprio_dim,), dtype=np.float32)

        return trajectories, transitions, {
            "mean": stacked_actions.mean(axis=0).astype(np.float32).tolist(),
            "std": stacked_actions.std(axis=0).astype(np.float32).tolist(),
            "min": stacked_actions.min(axis=0).astype(np.float32).tolist(),
            "max": stacked_actions.max(axis=0).astype(np.float32).tolist(),
            "q01": q01.tolist(),
            "q99": q99.tolist(),
            "mask": mask.tolist(),
            "proprio_mean": zero_proprio.tolist(),
            "proprio_std": zero_proprio.tolist(),
            "proprio_min": zero_proprio.tolist(),
            "proprio_max": zero_proprio.tolist(),
            "proprio_q01": zero_proprio.tolist(),
            "proprio_q99": zero_proprio.tolist(),
            "num_transitions": int(stacked_actions.shape[0]),
            "num_trajectories": len(trajectories),
        }

    @staticmethod
    def _standardize_actions(actions: np.ndarray) -> np.ndarray:
        standardized = np.asarray(actions, dtype=np.float32).copy()
        if standardized.ndim != 2 or standardized.shape[1] == 0:
            raise ValueError(f"Expected 2D action array with non-zero action dim, got {standardized.shape}")

        # Match OpenVLA's official LIBERO RLDS transform:
        # raw gripper action is -1=open, +1=close -> clip to [0, 1], then invert -> 1=open, 0=close.
        standardized[:, -1] = 1.0 - np.clip(standardized[:, -1], 0.0, 1.0)
        return standardized

    @staticmethod
    def _standardize_action(action: np.ndarray) -> np.ndarray:
        standardized = np.asarray(action, dtype=np.float32).copy()
        if standardized.ndim != 1 or standardized.shape[0] == 0:
            raise ValueError(f"Expected 1D action array with non-zero action dim, got {standardized.shape}")

        standardized[-1] = 1.0 - np.clip(standardized[-1], 0.0, 1.0)
        return standardized

    @staticmethod
    def _is_noop(action: np.ndarray, prev_action: np.ndarray | None, threshold: float = 1e-4) -> bool:
        if prev_action is None:
            return float(np.linalg.norm(action[:-1])) < threshold

        return float(np.linalg.norm(action[:-1])) < threshold and bool(action[-1] == prev_action[-1])

    def _get_kept_step_indices(self, actions: np.ndarray) -> tuple[int, ...]:
        kept_indices: List[int] = []
        prev_action: np.ndarray | None = None

        for step_idx, action in enumerate(np.asarray(actions, dtype=np.float32)):
            if self._is_noop(action, prev_action):
                continue

            kept_indices.append(step_idx)
            prev_action = action

        return tuple(kept_indices)

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        normalized = np.asarray(action, dtype=np.float32).copy()
        normalized[self._action_mask] = np.clip(
            2.0 * (action[self._action_mask] - self._action_q01[self._action_mask])
            / (self._action_q99[self._action_mask] - self._action_q01[self._action_mask] + 1e-8)
            - 1.0,
            -1.0,
            1.0,
        )
        normalized[self._zeros_mask] = 0.0
        return normalized

    def _prepare_image(self, image: np.ndarray) -> Image.Image:
        pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
        if self.image_aug:
            crop = RandomResizedCrop(
                size=pil_image.size[::-1],
                scale=(0.9, 0.9),
                ratio=(1.0, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            )
            pil_image = crop(pil_image)
            pil_image = ColorJitter(
                brightness=0.2,
                contrast=(0.8, 1.2),
                saturation=(0.8, 1.2),
                hue=0.05,
            )(pil_image)
        return pil_image

    def _iter_transition_indices(self, rng: np.random.Generator) -> Iterator[int]:
        num_transitions = len(self.transitions)

        if num_transitions == 0:
            return

        def next_transition_from_source(cursor: int) -> tuple[int, int | None]:
            if num_transitions == 0:
                return cursor, None

            if self.repeat:
                transition_idx = cursor % num_transitions
                return cursor + 1, transition_idx

            if cursor >= num_transitions:
                return cursor, None

            return cursor + 1, cursor

        source_cursor = 0
        buffer: List[int] = []
        # Mirror RLDS shuffle(buffer_size) more closely than trajectory-wise shuffling:
        # when `repeat=True` and the buffer is larger than one pass over the dataset,
        # the buffer will contain transitions from multiple repeated passes.
        target_buffer_size = self.shuffle_buffer_size if self.repeat else min(self.shuffle_buffer_size, num_transitions)
        while len(buffer) < target_buffer_size:
            source_cursor, transition_idx = next_transition_from_source(source_cursor)
            if transition_idx is None:
                break
            buffer.append(transition_idx)

        if not buffer:
            return

        while buffer:
            pick_idx = int(rng.integers(len(buffer)))
            transition_idx = buffer[pick_idx]
            yield transition_idx

            source_cursor, next_transition_idx = next_transition_from_source(source_cursor)
            if next_transition_idx is None:
                buffer[pick_idx] = buffer[-1]
                buffer.pop()
            else:
                buffer[pick_idx] = next_transition_idx

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        rng = np.random.default_rng(self.seed)
        handle_cache: Dict[str, h5py.File] = {}

        try:
            for transition_idx in self._iter_transition_indices(rng):
                transition = self.transitions[transition_idx]
                trajectory = self.trajectories[transition.trajectory_idx]
                if trajectory.file_path not in handle_cache:
                    handle_cache[trajectory.file_path] = h5py.File(trajectory.file_path, "r")

                demo = handle_cache[trajectory.file_path]["data"][trajectory.demo_key]
                image = self._prepare_image(np.asarray(demo["obs"][self.camera_view][transition.step_idx]))
                action = self._standardize_action(np.asarray(demo["actions"][transition.step_idx], dtype=np.float32))
                normalized_action = self._normalize_action(action)
                batch = {
                    "dataset_name": self.benchmark_name,
                    "action": normalized_action[None, ...],
                    "observation": {"image_primary": np.asarray(image)[None, ...]},
                    "task": {"language_instruction": trajectory.instruction.encode("utf-8")},
                }
                yield self.batch_transform(batch)
        finally:
            for handle in handle_cache.values():
                handle.close()

    def __len__(self) -> int:
        return self._num_transitions
