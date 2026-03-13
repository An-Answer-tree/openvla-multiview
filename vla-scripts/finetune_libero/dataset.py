import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info
from torchvision.transforms import ColorJitter, InterpolationMode, RandomResizedCrop


@dataclass(frozen=True)
class LiberoTrajectory:
    file_path: str
    demo_key: str
    instruction: str
    num_steps: int


class LiberoMultiviewDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        benchmark_name: str,
        batch_transform,
        camera_view: str = "agentview_rgb",
        train: bool = True,
        image_aug: bool = False,
        seed: int = 7,
        repeat: bool = True,
    ) -> None:
        self.data_root_dir = Path(data_root_dir)
        self.benchmark_name = benchmark_name
        self.batch_transform = batch_transform
        self.camera_view = camera_view
        self.train = train
        self.image_aug = image_aug
        self.seed = seed
        self.repeat = repeat

        self.dataset_dir = self._resolve_dataset_dir(self.data_root_dir, self.benchmark_name)
        self.trajectories, stats = self._index_dataset(self.dataset_dir)
        self._action_q01 = np.asarray(stats["q01"], dtype=np.float32)
        self._action_q99 = np.asarray(stats["q99"], dtype=np.float32)
        self._action_mask = np.asarray(stats["mask"], dtype=bool)
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

    def _index_dataset(self, dataset_dir: Path) -> tuple[List[LiberoTrajectory], Dict[str, Any]]:
        trajectories: List[LiberoTrajectory] = []
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

                    actions = np.asarray(demo["actions"], dtype=np.float32)
                    trajectories.append(
                        LiberoTrajectory(
                            file_path=str(hdf5_path),
                            demo_key=demo_key,
                            instruction=instruction,
                            num_steps=int(actions.shape[0]),
                        )
                    )
                    all_actions.append(actions)

        stacked_actions = np.concatenate(all_actions, axis=0)
        q01 = np.quantile(stacked_actions, 0.01, axis=0).astype(np.float32)
        q99 = np.quantile(stacked_actions, 0.99, axis=0).astype(np.float32)
        mask = (q99 > q01).astype(bool)

        return trajectories, {
            "mean": stacked_actions.mean(axis=0).astype(np.float32).tolist(),
            "std": stacked_actions.std(axis=0).astype(np.float32).tolist(),
            "min": stacked_actions.min(axis=0).astype(np.float32).tolist(),
            "max": stacked_actions.max(axis=0).astype(np.float32).tolist(),
            "q01": q01.tolist(),
            "q99": q99.tolist(),
            "mask": mask.tolist(),
            "num_transitions": int(stacked_actions.shape[0]),
            "num_trajectories": len(trajectories),
        }

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        normalized = np.zeros_like(action, dtype=np.float32)
        normalized[self._action_mask] = np.clip(
            2.0 * (action[self._action_mask] - self._action_q01[self._action_mask])
            / (self._action_q99[self._action_mask] - self._action_q01[self._action_mask] + 1e-8)
            - 1.0,
            -1.0,
            1.0,
        )
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

    def _iter_trajectory_indices(self, rng: np.random.Generator) -> Iterator[int]:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        num_shards = world_size * num_workers
        shard_id = rank * num_workers + worker_id

        while True:
            order = np.arange(len(self.trajectories))
            if self.train:
                rng.shuffle(order)

            for idx in order[shard_id::num_shards]:
                yield int(idx)

            if not self.repeat:
                return

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rank = int(os.environ.get("RANK", "0"))
        rng = np.random.default_rng(self.seed + 10_000 * rank + worker_id)

        for traj_idx in self._iter_trajectory_indices(rng):
            trajectory = self.trajectories[traj_idx]
            with h5py.File(trajectory.file_path, "r") as handle:
                demo = handle["data"][trajectory.demo_key]
                images = demo["obs"][self.camera_view]
                actions = np.asarray(demo["actions"], dtype=np.float32)

                step_order = np.arange(trajectory.num_steps)
                if self.train:
                    rng.shuffle(step_order)

                for step_idx in step_order:
                    image = self._prepare_image(np.asarray(images[step_idx]))
                    normalized_action = self._normalize_action(actions[step_idx])
                    batch = {
                        "dataset_name": self.benchmark_name,
                        "action": normalized_action[None, ...],
                        "observation": {"image_primary": np.asarray(image)[None, ...]},
                        "task": {"language_instruction": trajectory.instruction.encode("utf-8")},
                    }
                    yield self.batch_transform(batch)

    def __len__(self) -> int:
        return self._num_transitions
