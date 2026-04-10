"""
distillation.py

Minimal distillation entrypoint for LIBERO multiview datasets.
"""

import json
import os
import random
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import draccus
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from PIL import Image
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import IGNORE_INDEX, get_num_visual_tokens
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from camera_views import DEFAULT_CAMERA_NAMES, get_camera_views, resolve_camera_names
from dataset import LiberoMultiviewDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _default_student_target_camera_names() -> List[str]:
    return [camera_name for camera_name in DEFAULT_CAMERA_NAMES if camera_name != "agentview"]


@dataclass
class DistillationConfig:
    # fmt: off
    teacher_path: str = "openvla/openvla-7b"
    student_path: str = "openvla/openvla-7b"

    data_root_dir: Path = Path("/mnt/HDD-6940GB/Dataset/LIBERO-datasets-multiview-cameraINFO-double-backsideview")
    dataset_name: str = "libero_spatial"
    teacher_camera_names: List[str] = field(default_factory=lambda: list(DEFAULT_CAMERA_NAMES))
    student_target_camera_names: List[str] = field(default_factory=_default_student_target_camera_names)
    run_root_dir: Path = Path("runs")
    adapter_tmp_dir: Path = Path("adapter-tmp")

    batch_size: int = 1
    max_steps: int = 200_000
    save_steps: int = 5_000
    learning_rate: float = 5e-4
    grad_accumulation_steps: int = 16
    image_aug: bool = True
    shuffle_buffer_size: int = 100_000
    seed: int = 7
    global_shuffle_across_ranks: bool = True
    save_latest_checkpoint_only: bool = False

    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    use_quantization: bool = False
    adapter_hidden_scale: float = 4.0

    action_loss_weight: float = 1.0
    distill_patch_loss_weight: float = 1.0
    distill_projector_loss_weight: float = 1.0

    student_gpu_ids: List[int] = field(default_factory=lambda: [0])
    teacher_gpu_ids: List[int] = field(default_factory=lambda: [0])

    wandb_project: str = "distillation"
    wandb_entity: str = "szliutong-wuhan-university"
    run_id_note: Optional[str] = None
    # fmt: on


@dataclass
class DistillationBatchTransform:
    """Builds token labels and pixel values for distillation training."""

    action_tokenizer: ActionTokenizer
    base_tokenizer: Any
    image_transform: Any
    prompt_builder_fn: Any
    predict_stop_token: bool = True

    def build_language_tensors(self, action: np.ndarray, instruction: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Builds tokenized prompt and labels for one transition."""
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        input_ids_t = torch.tensor(input_ids)
        labels_t = torch.tensor(labels)
        labels_t[: -(len(action) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels_t[-1] = IGNORE_INDEX

        return input_ids_t, labels_t

    def transform_image(self, image: np.ndarray) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Applies the OpenVLA image transform to one RGB image."""
        pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
        return self.image_transform(pil_image)

    def transform_images(
        self, images: Sequence[np.ndarray]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Applies the image transform to one or more RGB images."""
        transformed_images = [self.transform_image(image) for image in images]
        first_transformed_image = transformed_images[0]

        if len(transformed_images) == 1:
            return first_transformed_image

        if isinstance(first_transformed_image, torch.Tensor):
            return torch.stack(transformed_images)

        if isinstance(first_transformed_image, dict):
            return {
                key: torch.stack([transformed_image[key] for transformed_image in transformed_images])
                for key in first_transformed_image
            }

        raise ValueError(f"Unsupported transformed image type `{type(first_transformed_image)}`")


class LiberoDistillationDataset(LiberoMultiviewDataset):
    """Dataset that returns teacher and student views for the same transition."""

    def __init__(
        self,
        data_root_dir: Path,
        benchmark_name: str,
        batch_transform: DistillationBatchTransform,
        teacher_camera_views: Sequence[str],
        student_camera_views: Sequence[str],
        train: bool = True,
        image_aug: bool = False,
        shuffle_buffer_size: int = 100_000,
        seed: Optional[int] = None,
        repeat: bool = True,
        global_shuffle_across_ranks: bool = False,
    ) -> None:
        all_camera_views = tuple(dict.fromkeys([*teacher_camera_views, *student_camera_views]))
        super().__init__(
            data_root_dir=data_root_dir,
            benchmark_name=benchmark_name,
            batch_transform=batch_transform,
            camera_views=all_camera_views,
            train=train,
            image_aug=image_aug,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
            repeat=repeat,
            global_shuffle_across_ranks=global_shuffle_across_ranks,
        )
        self.distillation_batch_transform = batch_transform
        self.teacher_camera_views = tuple(teacher_camera_views)
        self.student_camera_views = tuple(student_camera_views)

        if not self.teacher_camera_views:
            raise ValueError("Expected at least one teacher camera view.")
        if not self.student_camera_views:
            raise ValueError("Expected at least one student camera view.")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rank = int(os.environ.get("RANK", "0"))
        if self.seed is None:
            rng = np.random.default_rng()
        elif self.global_shuffle_across_ranks:
            rng = np.random.default_rng(self.seed)
        else:
            rng = np.random.default_rng(self.seed + 10_000 * rank + worker_id)
        handle_cache: Dict[str, Any] = {}

        try:
            for transition_idx in self._iter_sharded_transition_indices(rng):
                transition = self.transitions[transition_idx]
                trajectory = self.trajectories[transition.trajectory_idx]
                if trajectory.file_path not in handle_cache:
                    import h5py

                    handle_cache[trajectory.file_path] = h5py.File(trajectory.file_path, "r")

                demo = handle_cache[trajectory.file_path]["data"][trajectory.demo_key]
                student_view_index = int(rng.integers(len(self.student_camera_views)))
                student_camera_view = self.student_camera_views[student_view_index]

                teacher_images = [
                    np.asarray(
                        self._prepare_image(np.asarray(demo["obs"][camera_view][transition.step_idx])),
                        dtype=np.uint8,
                    )
                    for camera_view in self.teacher_camera_views
                ]
                student_image = np.asarray(
                    self._prepare_image(np.asarray(demo["obs"][student_camera_view][transition.step_idx])),
                    dtype=np.uint8,
                )

                action = self._standardize_action(np.asarray(demo["actions"][transition.step_idx], dtype=np.float32))
                normalized_action = self._normalize_action(action)
                input_ids, labels = self.distillation_batch_transform.build_language_tensors(
                    normalized_action,
                    trajectory.instruction,
                )

                yield {
                    "teacher_pixel_values": self.distillation_batch_transform.transform_images(teacher_images),
                    "student_pixel_values": self.distillation_batch_transform.transform_image(student_image),
                    "student_view_index": student_view_index,
                    "input_ids": input_ids,
                    "labels": labels,
                    "dataset_name": self.benchmark_name,
                }
        finally:
            for handle in handle_cache.values():
                handle.close()


@dataclass
class DistillationCollator:
    """Pads token tensors and stacks teacher/student pixel values."""

    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"

    def _stack_pixel_values(
        self, pixel_values: Sequence[Union[torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        first_pixel_values = pixel_values[0]
        if isinstance(first_pixel_values, torch.Tensor):
            return torch.stack(list(pixel_values))
        if isinstance(first_pixel_values, dict):
            return {
                key: torch.stack([instance_pixel_values[key] for instance_pixel_values in pixel_values])
                for key in first_pixel_values
            }
        raise ValueError(f"Unsupported `pixel_values` type = {type(first_pixel_values)}")

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Collates one distillation batch."""
        assert self.padding_side == "right", f"Invalid tokenizer `{self.padding_side = }`"
        input_ids = pad_sequence(
            [instance["input_ids"] for instance in instances],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        labels = pad_sequence(
            [instance["labels"] for instance in instances],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        input_ids = input_ids[:, : self.model_max_length]
        labels = labels[:, : self.model_max_length]
        attention_mask = input_ids.ne(self.pad_token_id)

        return {
            "teacher_pixel_values": self._stack_pixel_values([instance["teacher_pixel_values"] for instance in instances]),
            "student_pixel_values": self._stack_pixel_values([instance["student_pixel_values"] for instance in instances]),
            "student_view_indices": torch.tensor(
                [instance["student_view_index"] for instance in instances],
                dtype=torch.long,
            ),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "dataset_names": [instance["dataset_name"] for instance in instances],
        }


class ResidualViewAdapter(nn.Module):
    """Residual MLP adapter applied to one target view."""

    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class DistillationStudentModel(nn.Module):
    """Student model with view-specific visual adapters."""

    def __init__(self, vla: nn.Module, student_camera_names: Sequence[str], adapter_hidden_scale: float) -> None:
        super().__init__()
        self.vla = vla
        self.student_camera_names = tuple(student_camera_names)

        base_model = self._get_base_model()
        feature_dim = int(base_model.vision_backbone.embed_dim)
        hidden_dim = max(int(round(feature_dim * adapter_hidden_scale)), feature_dim)
        self.view_adapters = nn.ModuleDict(
            {
                camera_name: ResidualViewAdapter(feature_dim=feature_dim, hidden_dim=hidden_dim)
                for camera_name in self.student_camera_names
            }
        )

    def _get_base_model(self) -> nn.Module:
        if hasattr(self.vla, "get_base_model"):
            return self.vla.get_base_model()
        return self.vla

    def _apply_view_adapters(self, vision_features: torch.Tensor, student_view_indices: torch.Tensor) -> torch.Tensor:
        adapted_features = vision_features.clone()
        for view_index, camera_name in enumerate(self.student_camera_names):
            mask = student_view_indices == view_index
            if not torch.any(mask):
                continue
            adapted_features[mask] = self.view_adapters[camera_name](vision_features[mask])
        return adapted_features

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: torch.Tensor,
        student_view_indices: torch.Tensor,
    ) -> CausalLMOutputWithPast:
        base_model = self._get_base_model()
        vision_features = base_model.get_vision_features(pixel_values)
        adapted_vision_features = self._apply_view_adapters(vision_features, student_view_indices)
        return base_model.forward_from_vision_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vision_features=adapted_vision_features,
            labels=labels,
            output_projector_features=True,
            output_vision_features=True,
            return_dict=True,
        )


def _move_pixel_values_to_device(
    pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]],
    device: torch.device,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """Moves a pixel-values structure to the target device in BF16."""
    if isinstance(pixel_values, torch.Tensor):
        return pixel_values.to(torch.bfloat16).to(device)
    if isinstance(pixel_values, dict):
        return {key: value.to(torch.bfloat16).to(device) for key, value in pixel_values.items()}
    raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")


def _mean_pool_features(features: torch.Tensor) -> torch.Tensor:
    """Pools visual tokens into one global feature vector."""
    if features.ndim != 3:
        raise ValueError(f"Expected feature tensor rank 3, got shape {tuple(features.shape)}")
    return features.mean(dim=1)


def _to_cpu_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    """Builds a CPU state dict for serialization."""
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def _serialize_cfg(cfg: DistillationConfig) -> Dict[str, Any]:
    """Converts config values into a JSON-serializable dictionary."""
    serialized = {}
    for key, value in vars(cfg).items():
        if isinstance(value, Path):
            serialized[key] = str(value)
        else:
            serialized[key] = value
    return serialized


def _get_rank_device_ids(cfg: DistillationConfig, rank: int, world_size: int) -> tuple[int, int]:
    """Returns the configured student/teacher GPU ids for one rank."""
    if len(cfg.student_gpu_ids) != world_size:
        raise ValueError(
            f"Expected `student_gpu_ids` to have length {world_size}, got {len(cfg.student_gpu_ids)}"
        )
    if len(cfg.teacher_gpu_ids) != world_size:
        raise ValueError(
            f"Expected `teacher_gpu_ids` to have length {world_size}, got {len(cfg.teacher_gpu_ids)}"
        )
    if len(set(cfg.teacher_gpu_ids)) != len(cfg.teacher_gpu_ids):
        raise ValueError("Teacher GPU ids must be unique across ranks in the minimal implementation.")

    available_device_count = torch.cuda.device_count()
    for device_id in [*cfg.student_gpu_ids, *cfg.teacher_gpu_ids]:
        if device_id < 0 or device_id >= available_device_count:
            raise ValueError(
                f"Configured CUDA device id {device_id} is invalid for {available_device_count} visible GPUs."
            )

    return cfg.student_gpu_ids[rank], cfg.teacher_gpu_ids[rank]


def _build_experiment_id(
    cfg: DistillationConfig,
    teacher_camera_names: Sequence[str],
    student_camera_names: Sequence[str],
) -> str:
    """Builds a readable experiment identifier."""
    exp_id_parts = [
        "distillation",
        cfg.dataset_name,
        f"teacher-{_summarize_teacher_cameras(teacher_camera_names)}",
        f"student-{len(student_camera_names)}v",
        datetime.now().strftime("%m%d-%H%M"),
    ]
    return "+".join(exp_id_parts)


def _summarize_teacher_cameras(camera_names: Sequence[str]) -> str:
    """Returns a compact identifier for the teacher camera setup."""
    if len(camera_names) == 1:
        return camera_names[0]
    return f"{len(camera_names)}v"


def _save_distillation_checkpoint(
    cfg: DistillationConfig,
    step: int,
    run_dir: Path,
    adapter_dir: Path,
    processor: Any,
    dataset_statistics: Dict[str, Any],
    student_model: DistillationStudentModel,
) -> None:
    """Saves one distilled student checkpoint."""
    checkpoint_dir = run_dir if cfg.save_latest_checkpoint_only else Path(str(run_dir) + f"--{step}_chkpt")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_dataset_statistics(dataset_statistics, checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)

    metadata = {
        "teacher_camera_names": list(resolve_camera_names(cfg.teacher_camera_names)),
        "student_target_camera_names": list(resolve_camera_names(cfg.student_target_camera_names)),
        "adapter_hidden_scale": cfg.adapter_hidden_scale,
        "config": _serialize_cfg(cfg),
    }
    with open(checkpoint_dir / "distillation_config.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    torch.save(_to_cpu_state_dict(student_model.view_adapters), checkpoint_dir / "view_adapters.pt")

    if cfg.use_lora:
        student_model.vla.save_pretrained(adapter_dir)
        base_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.student_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir).merge_and_unload()
        merged_vla.save_pretrained(checkpoint_dir)
    else:
        student_model.vla.save_pretrained(checkpoint_dir)


def _compute_feature_distillation_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    use_token_level_loss: bool,
) -> torch.Tensor:
    """Computes an MSE distillation loss over token or pooled features."""
    if use_token_level_loss:
        if student_features.shape != teacher_features.shape:
            raise ValueError(
                "Token-level distillation requires matching feature shapes, got "
                f"{tuple(student_features.shape)} vs {tuple(teacher_features.shape)}"
            )
        return F.mse_loss(student_features.float(), teacher_features.float())

    if student_features.ndim != 3:
        raise ValueError(
            "Expected student features for pooled distillation to have rank 3, got "
            f"{student_features.ndim} with shape {tuple(student_features.shape)}"
        )
    if teacher_features.ndim != 2:
        raise ValueError(
            "Expected teacher features for pooled distillation to have rank 2, got "
            f"{teacher_features.ndim} with shape {tuple(teacher_features.shape)}"
        )

    pooled_student_features = _mean_pool_features(student_features).float()
    pooled_teacher_features = teacher_features.float()
    return F.mse_loss(pooled_student_features, pooled_teacher_features)


@draccus.wrap()
def distill(cfg: DistillationConfig) -> None:
    teacher_camera_names = resolve_camera_names(cfg.teacher_camera_names)
    student_target_camera_names = resolve_camera_names(cfg.student_target_camera_names)
    teacher_camera_views = get_camera_views(teacher_camera_names)
    student_camera_views = get_camera_views(student_target_camera_names)

    print(
        f"Distilling OpenVLA student `{cfg.student_path}` from teacher `{cfg.teacher_path}` "
        f"with teacher cameras `{list(teacher_camera_names)}` and student targets `{list(student_target_camera_names)}`"
    )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("Distillation assumes at least one GPU is available.")

    distributed_state = PartialState()
    world_size = distributed_state.num_processes
    rank = distributed_state.process_index
    student_device_id, teacher_device_id = _get_rank_device_ids(cfg, rank, world_size)
    student_device = torch.device(f"cuda:{student_device_id}")
    teacher_device = torch.device(f"cuda:{teacher_device_id}")
    torch.cuda.set_device(student_device_id)
    torch.cuda.empty_cache()

    exp_id = _build_experiment_id(cfg, teacher_camera_names, student_target_camera_names)
    run_dir = cfg.run_root_dir / exp_id
    adapter_dir = cfg.adapter_tmp_dir / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)

    quantization_config = None
    if cfg.use_quantization:
        if not cfg.use_lora:
            raise ValueError("Quantized distillation is only supported together with LoRA.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(cfg.student_path)
    teacher = AutoModelForVision2Seq.from_pretrained(
        cfg.teacher_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(teacher_device)
    teacher.eval()
    teacher.requires_grad_(False)

    student_vla = AutoModelForVision2Seq.from_pretrained(
        cfg.student_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
    )
    if cfg.use_quantization:
        student_vla = prepare_model_for_kbit_training(student_vla)
    else:
        student_vla = student_vla.to(student_device)

    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        student_vla = get_peft_model(student_vla, lora_config)
        student_vla.print_trainable_parameters()

    student_model = DistillationStudentModel(
        vla=student_vla,
        student_camera_names=student_target_camera_names,
        adapter_hidden_scale=cfg.adapter_hidden_scale,
    )
    if cfg.use_quantization:
        student_model.view_adapters = student_model.view_adapters.to(student_device)
    else:
        student_model = student_model.to(student_device)

    use_ddp = world_size > 1
    if use_ddp:
        student_model = DDP(
            student_model,
            device_ids=[student_device_id],
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )

    student_core = student_model.module if use_ddp else student_model
    optimizer = AdamW([parameter for parameter in student_model.parameters() if parameter.requires_grad], lr=cfg.learning_rate)

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = DistillationBatchTransform(
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.student_path else VicunaV15ChatPromptBuilder,
    )
    distillation_dataset = LiberoDistillationDataset(
        data_root_dir=cfg.data_root_dir,
        benchmark_name=cfg.dataset_name,
        batch_transform=batch_transform,
        teacher_camera_views=teacher_camera_views,
        student_camera_views=student_camera_views,
        train=True,
        image_aug=cfg.image_aug,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        seed=cfg.seed,
        global_shuffle_across_ranks=cfg.global_shuffle_across_ranks,
    )

    if distributed_state.is_main_process:
        save_dataset_statistics(distillation_dataset.dataset_statistics, run_dir)
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"distill+{exp_id}")

    collator = DistillationCollator(
        model_max_length=processor.tokenizer.model_max_length,
        pad_token_id=processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    dataloader = DataLoader(
        distillation_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )

    recent_total_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_patch_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_projector_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    teacher_uses_token_level_loss = len(teacher_camera_names) == 1

    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        student_model.train()
        teacher.eval()
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            teacher_pixel_values = _move_pixel_values_to_device(batch["teacher_pixel_values"], teacher_device)
            student_pixel_values = _move_pixel_values_to_device(batch["student_pixel_values"], student_device)
            input_ids = batch["input_ids"].to(student_device)
            attention_mask = batch["attention_mask"].to(student_device)
            labels = batch["labels"].to(student_device)
            student_view_indices = batch["student_view_indices"].to(student_device)

            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    teacher_vision_features = teacher.get_vision_features(teacher_pixel_values)
                    teacher_projector_features = teacher.get_projector_features(teacher_vision_features)

                if teacher_uses_token_level_loss:
                    teacher_patch_targets = teacher_vision_features.to(student_device)
                    teacher_projector_targets = teacher_projector_features.to(student_device)
                else:
                    teacher_patch_targets = _mean_pool_features(teacher_vision_features).to(student_device)
                    teacher_projector_targets = _mean_pool_features(teacher_projector_features).to(student_device)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                student_output: CausalLMOutputWithPast = student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=student_pixel_values,
                    labels=labels,
                    student_view_indices=student_view_indices,
                )
                action_loss = student_output.loss

                patch_distill_loss = _compute_feature_distillation_loss(
                    student_features=student_output.vision_features,
                    teacher_features=teacher_patch_targets,
                    use_token_level_loss=teacher_uses_token_level_loss,
                )
                projector_distill_loss = _compute_feature_distillation_loss(
                    student_features=student_output.projector_features,
                    teacher_features=teacher_projector_targets,
                    use_token_level_loss=teacher_uses_token_level_loss,
                )

                total_loss = (
                    cfg.action_loss_weight * action_loss
                    + cfg.distill_patch_loss_weight * patch_distill_loss
                    + cfg.distill_projector_loss_weight * projector_distill_loss
                )

            normalized_loss = total_loss / cfg.grad_accumulation_steps
            normalized_loss.backward()

            student_base_model = student_core._get_base_model()
            num_visual_tokens = get_num_visual_tokens(
                batch["student_pixel_values"],
                student_base_model.vision_backbone.num_patches,
            )
            action_logits = student_output.logits[:, num_visual_tokens:-1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = labels[:, 1:]
            mask = action_gt > action_tokenizer.action_token_begin_idx
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].detach().cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].detach().cpu().numpy())
            )
            action_l1_loss = F.l1_loss(continuous_actions_pred, continuous_actions_gt)

            recent_total_losses.append(total_loss.item())
            recent_action_losses.append(action_loss.item())
            recent_patch_losses.append(patch_distill_loss.item())
            recent_projector_losses.append(projector_distill_loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())

            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            smooth_total_loss = sum(recent_total_losses) / len(recent_total_losses)
            smooth_action_loss = sum(recent_action_losses) / len(recent_action_losses)
            smooth_patch_loss = sum(recent_patch_losses) / len(recent_patch_losses)
            smooth_projector_loss = sum(recent_projector_losses) / len(recent_projector_losses)
            smooth_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
            smooth_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

            if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                wandb.log(
                    {
                        "train_loss": smooth_total_loss,
                        "action_loss": smooth_action_loss,
                        "patch_distill_loss": smooth_patch_loss,
                        "projector_distill_loss": smooth_projector_loss,
                        "action_accuracy": smooth_action_accuracy,
                        "l1_loss": smooth_l1_loss,
                    },
                    step=gradient_step_idx,
                )

            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                progress.update()

            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                if distributed_state.is_main_process:
                    print(f"Saving Distillation Checkpoint for Step {gradient_step_idx}")
                    _save_distillation_checkpoint(
                        cfg=cfg,
                        step=gradient_step_idx,
                        run_dir=run_dir,
                        adapter_dir=adapter_dir,
                        processor=processor,
                        dataset_statistics=distillation_dataset.dataset_statistics,
                        student_model=student_core,
                    )

                if use_ddp:
                    dist.barrier()

            if gradient_step_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    distill()
