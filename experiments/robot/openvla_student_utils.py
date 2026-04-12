"""Utilities for evaluating distilled OpenVLA student checkpoints."""

from __future__ import annotations

import contextlib
import json
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from experiments.robot.libero.libero_utils import resolve_camera_names
from experiments.robot.openvla_utils import DEVICE, OPENVLA_V01_SYSTEM_PROMPT, _center_crop_pil_image, get_vla


@dataclass(frozen=True)
class DistillationStudentMetadata:
    """Metadata stored alongside a distilled student checkpoint.

    Attributes:
      teacher_camera_names: Teacher cameras used during distillation.
      student_target_camera_names: Student cameras with dedicated adapters.
      adapter_hidden_scale: Adapter expansion ratio from the checkpoint metadata.
    """

    teacher_camera_names: tuple[str, ...]
    student_target_camera_names: tuple[str, ...]
    adapter_hidden_scale: float


class ResidualViewAdapter(nn.Module):
    """Residual MLP adapter applied to one student camera view."""

    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        """Initializes the residual adapter.

        Args:
          feature_dim: Input and output feature width.
          hidden_dim: Hidden layer width inside the adapter MLP.
        """
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies one residual adapter block."""
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class StudentVLAInference:
    """Inference wrapper that injects view-specific adapters into OpenVLA."""

    def __init__(
        self,
        vla: nn.Module,
        metadata: DistillationStudentMetadata,
        view_adapters: nn.ModuleDict,
    ) -> None:
        """Initializes the student inference wrapper.

        Args:
          vla: Merged OpenVLA checkpoint saved by distillation training.
          metadata: Distillation metadata loaded from checkpoint files.
          view_adapters: Per-view adapter modules restored from checkpoint.
        """
        self.vla = vla
        self.norm_stats = vla.norm_stats
        self.teacher_camera_names = metadata.teacher_camera_names
        self.student_target_camera_names = metadata.student_target_camera_names
        self.view_adapters = view_adapters
        self.vla.eval()
        self.view_adapters.eval()

    def predict_action(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        unnorm_key: Optional[str] = None,
        student_camera_name: Optional[str] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Runs OpenVLA action prediction with the chosen student adapter.

        Args:
          input_ids: Tokenized prompt ids.
          unnorm_key: Dataset key used for action un-normalization.
          student_camera_name: Logical camera name whose adapter should be used.
          **kwargs: Extra keyword arguments forwarded to `OpenVLA.predict_action`.

        Returns:
          The predicted continuous action.

        Raises:
          ValueError: If the requested camera does not exist in the checkpoint.
        """
        if student_camera_name is None:
            raise ValueError("Expected `student_camera_name` for distilled student inference.")
        if student_camera_name not in self.student_target_camera_names:
            raise ValueError(
                "Unsupported student eval camera "
                f"`{student_camera_name}`. Choose from: {list(self.student_target_camera_names)}"
            )

        with self._use_view_adapter(student_camera_name):
            return self.vla.predict_action(input_ids=input_ids, unnorm_key=unnorm_key, **kwargs)

    @contextlib.contextmanager
    def _use_view_adapter(self, student_camera_name: str) -> Iterator[None]:
        """Temporarily patches vision feature extraction with one adapter."""
        original_get_vision_features = self.vla.get_vision_features
        adapter = self.view_adapters[student_camera_name]

        def patched_get_vision_features(model_self: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
            """Applies the selected student adapter to visual patch features."""
            vision_features = original_get_vision_features(pixel_values)
            adapter_parameter = next(adapter.parameters())
            adapted_input = vision_features.to(device=adapter_parameter.device, dtype=adapter_parameter.dtype)
            adapted_features = adapter(adapted_input)
            return adapted_features.to(device=vision_features.device, dtype=vision_features.dtype)

        self.vla.get_vision_features = types.MethodType(patched_get_vision_features, self.vla)
        try:
            yield
        finally:
            self.vla.get_vision_features = original_get_vision_features


def load_student_vla(cfg: Any) -> StudentVLAInference:
    """Loads a distilled student checkpoint for evaluation.

    Args:
      cfg: Eval config object that provides checkpoint and load options.

    Returns:
      A wrapped OpenVLA student ready for single-view evaluation.
    """
    checkpoint_dir = Path(cfg.pretrained_checkpoint).expanduser()
    metadata = load_distillation_student_metadata(checkpoint_dir)
    vla = get_vla(cfg)
    view_adapters = _build_view_adapters(vla, metadata, checkpoint_dir)
    return StudentVLAInference(vla=vla, metadata=metadata, view_adapters=view_adapters)


def get_student_vla_action(
    model: StudentVLAInference,
    processor: Any,
    base_vla_name: str,
    obs: dict[str, Any],
    task_label: str,
    unnorm_key: str,
    student_camera_name: str,
    center_crop: bool = False,
) -> np.ndarray:
    """Generates an action with a distilled student checkpoint.

    Args:
      model: Distilled student inference wrapper.
      processor: Hugging Face processor loaded from the checkpoint.
      base_vla_name: Checkpoint identifier used to choose the prompt format.
      obs: Eval observation dictionary with image inputs.
      task_label: Language task description.
      unnorm_key: Dataset key used for action un-normalization.
      student_camera_name: Camera name that selects the active student adapter.
      center_crop: Whether to apply the evaluation-time center crop.

    Returns:
      A predicted continuous action vector.
    """
    image = _get_student_input_image(obs, center_crop=center_crop)
    prompt = _build_vla_prompt(base_vla_name, task_label)
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
    return model.predict_action(
        **inputs,
        unnorm_key=unnorm_key,
        student_camera_name=student_camera_name,
        do_sample=False,
    )


def load_distillation_student_metadata(pretrained_checkpoint: Union[str, Path]) -> DistillationStudentMetadata:
    """Loads and validates distilled student metadata from checkpoint files."""
    checkpoint_dir = Path(pretrained_checkpoint).expanduser()
    metadata_path = checkpoint_dir / "distillation_config.json"
    adapter_weights_path = checkpoint_dir / "view_adapters.pt"

    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Distilled student checkpoint is missing `distillation_config.json`: "
            f"{metadata_path}"
        )
    if not adapter_weights_path.is_file():
        raise FileNotFoundError(
            "Distilled student checkpoint is missing `view_adapters.pt`: "
            f"{adapter_weights_path}"
        )

    with open(metadata_path, "r", encoding="utf-8") as handle:
        raw_metadata = json.load(handle)

    try:
        teacher_camera_names = resolve_camera_names(raw_metadata["teacher_camera_names"])
        student_target_camera_names = resolve_camera_names(raw_metadata["student_target_camera_names"])
        adapter_hidden_scale = float(raw_metadata["adapter_hidden_scale"])
    except KeyError as exc:
        raise ValueError(
            "Distilled student metadata is missing a required key in "
            f"{metadata_path}: {exc}"
        ) from exc

    return DistillationStudentMetadata(
        teacher_camera_names=teacher_camera_names,
        student_target_camera_names=student_target_camera_names,
        adapter_hidden_scale=adapter_hidden_scale,
    )


def _build_view_adapters(
    vla: nn.Module,
    metadata: DistillationStudentMetadata,
    checkpoint_dir: Path,
) -> nn.ModuleDict:
    """Constructs view adapters that match the distillation training layout."""
    feature_dim = int(vla.vision_backbone.embed_dim)
    hidden_dim = max(int(round(feature_dim * metadata.adapter_hidden_scale)), feature_dim)
    view_adapters = nn.ModuleDict(
        {
            camera_name: ResidualViewAdapter(feature_dim=feature_dim, hidden_dim=hidden_dim)
            for camera_name in metadata.student_target_camera_names
        }
    )
    adapter_state_dict = torch.load(checkpoint_dir / "view_adapters.pt", map_location="cpu")
    try:
        view_adapters.load_state_dict(adapter_state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "Failed to load student view adapters from "
            f"{checkpoint_dir / 'view_adapters.pt'}"
        ) from exc

    projector_parameter = next(vla.projector.parameters())
    view_adapters = view_adapters.to(device=projector_parameter.device, dtype=projector_parameter.dtype)
    return view_adapters


def _get_student_input_image(obs: dict[str, Any], center_crop: bool = False) -> Image.Image:
    """Builds one PIL image for single-view student evaluation."""
    if "full_images" in obs:
        raw_images = np.asarray(obs["full_images"])
        if raw_images.shape[0] != 1:
            raise ValueError(
                "Distilled student eval expects exactly one input view, got "
                f"{raw_images.shape[0]} views."
            )
        image = Image.fromarray(np.asarray(raw_images[0], dtype=np.uint8)).convert("RGB")
    elif "full_image" in obs:
        image = Image.fromarray(np.asarray(obs["full_image"], dtype=np.uint8)).convert("RGB")
    else:
        raise KeyError("Expected observation to contain `full_image` or `full_images`.")

    if center_crop:
        image = _center_crop_pil_image(image)

    return image


def _build_vla_prompt(base_vla_name: str, task_label: str) -> str:
    """Formats the text prompt used for OpenVLA action prediction."""
    if "openvla-v01" in base_vla_name:
        return (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to "
            f"{task_label.lower()}? ASSISTANT:"
        )
    return f"In: What action should the robot take to {task_label.lower()}?\nOut:"
