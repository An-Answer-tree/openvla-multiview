"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import wraps
from typing import Iterable, Optional, Sequence

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import imageio
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from scipy.spatial.transform import Rotation

from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)


DEFAULT_OPERATION_CAMERA_BASE_NAME = "agentview"
DEFAULT_OPERATION_CAMERA_NAMES = {
    "top": "operation_topview",
    "left": "operation_leftview",
    "right": "operation_rightview",
    "back": "operation_backview",
    "left_back": "operation_leftbackview",
    "right_back": "operation_rightbackview",
}
SUPPORTED_CAMERA_NAMES = (
    "agentview",
    "topview",
    "leftview",
    "rightview",
    "backview",
    "leftbackview",
    "rightbackview",
)
DEFAULT_CAMERA_NAMES = SUPPORTED_CAMERA_NAMES
FRONT_CAMERA_NAME = "frontview"
CAMERA_NAME_TO_ENV_NAME = {
    "agentview": "agentview",
    "topview": "operation_topview",
    "leftview": "operation_leftview",
    "rightview": "operation_rightview",
    "backview": "operation_backview",
    "leftbackview": "operation_leftbackview",
    "rightbackview": "operation_rightbackview",
}
CAMERA_NAME_TO_OBS_KEY = {
    camera_name: f"{env_camera_name}_image"
    for camera_name, env_camera_name in CAMERA_NAME_TO_ENV_NAME.items()
}

_MULTIVIEW_CAMERA_SETUP_HOOKS_INSTALLED = False


@dataclass(frozen=True)
class OperationCameraConfig:
    """Configuration for generated fixed operation cameras."""

    base_camera_name: str = DEFAULT_OPERATION_CAMERA_BASE_NAME
    camera_names: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_OPERATION_CAMERA_NAMES)
    )


@dataclass(frozen=True)
class CameraSpec:
    """Description of a camera node injected into the MuJoCo arena."""

    name: str
    pos: np.ndarray
    quat: np.ndarray
    mode: str = "fixed"
    fovy: Optional[str] = None


def _dedupe_keep_order(values):
    """Returns a list without duplicates while preserving the input order."""
    return list(dict.fromkeys(values))


def _parse_vector_attr(attr_value: Optional[str], dim: int, attr_name: str) -> np.ndarray:
    """Parses a whitespace-separated XML attribute into a float vector."""

    if attr_value is None:
        raise ValueError(f"Missing camera attribute `{attr_name}` in XML")
    values = np.asarray([float(value) for value in attr_value.split()], dtype=np.float64)
    if values.shape[0] != dim:
        raise ValueError(
            f"Camera attribute `{attr_name}` must have {dim} numbers, got {values.shape[0]}"
        )
    return values


def _camera_pose_from_element(
    camera_elem: ET.Element,
) -> tuple[np.ndarray, np.ndarray, Optional[str]]:
    """Returns ``(pos, quat, fovy)`` from a camera XML element."""

    pos = _parse_vector_attr(camera_elem.get("pos"), 3, "pos")
    quat = _parse_vector_attr(camera_elem.get("quat", "1 0 0 0"), 4, "quat")
    return pos, quat, camera_elem.get("fovy")


def _normalize(vec: np.ndarray) -> np.ndarray:
    """Normalizes a vector and raises if the vector length is zero."""

    norm = np.linalg.norm(vec)
    if norm <= 1e-12:
        raise ValueError("Cannot normalize zero-length vector")
    return vec / norm


def _rotate_xy(vec: np.ndarray, degrees: float) -> np.ndarray:
    """Rotates a vector around the world z-axis in the x-y plane."""

    radians = np.deg2rad(degrees)
    cos_v = np.cos(radians)
    sin_v = np.sin(radians)
    return np.array(
        [
            cos_v * vec[0] - sin_v * vec[1],
            sin_v * vec[0] + cos_v * vec[1],
            vec[2],
        ],
        dtype=np.float64,
    )


def _lookat_quat_wxyz(camera_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    """Constructs a MuJoCo wxyz quaternion for a look-at camera pose."""

    forward = _normalize(target_pos - camera_pos)
    z_axis = -forward

    up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_axis = np.cross(up_hint, z_axis)
    if np.linalg.norm(x_axis) <= 1e-8:
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis = np.cross(up_hint, z_axis)

    x_axis = _normalize(x_axis)
    y_axis = _normalize(np.cross(z_axis, x_axis))
    rot = np.column_stack([x_axis, y_axis, z_axis])

    quat_xyzw = Rotation.from_matrix(rot).as_quat()
    return np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float64,
    )


def _pitch_target_up(
    camera_pos: np.ndarray,
    target_pos: np.ndarray,
    degrees: float,
) -> np.ndarray:
    """Rotates a camera target upward around the camera-local right axis."""

    direction = np.asarray(target_pos, dtype=np.float64) - np.asarray(
        camera_pos, dtype=np.float64
    )
    distance = float(np.linalg.norm(direction))
    if distance <= 1e-12:
        raise ValueError("Cannot pitch a zero-length camera direction")

    direction = direction / distance
    up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right_axis = np.cross(direction, up_hint)
    if np.linalg.norm(right_axis) <= 1e-8:
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right_axis = np.cross(direction, up_hint)
    right_axis = _normalize(right_axis)

    pitched_direction = Rotation.from_rotvec(
        np.deg2rad(degrees) * right_axis
    ).apply(direction)
    return np.asarray(camera_pos, dtype=np.float64) + pitched_direction * distance


def _advance_along_view(
    camera_pos: np.ndarray,
    target_pos: np.ndarray,
    distance: float,
) -> np.ndarray:
    """Moves a camera forward along its current view direction."""

    return np.asarray(camera_pos, dtype=np.float64) + _normalize(
        np.asarray(target_pos, dtype=np.float64) - np.asarray(camera_pos, dtype=np.float64)
    ) * float(distance)


def _camera_center_from_pose(base_pos: np.ndarray, base_quat_wxyz: np.ndarray) -> np.ndarray:
    """Approximates the look-at center from a camera world pose."""

    quat_xyzw = np.array(
        [base_quat_wxyz[1], base_quat_wxyz[2], base_quat_wxyz[3], base_quat_wxyz[0]],
        dtype=np.float64,
    )
    base_forward = Rotation.from_quat(quat_xyzw).apply(np.array([0.0, 0.0, -1.0]))
    base_radius = float(np.linalg.norm(base_pos))
    if base_radius <= 1e-12:
        base_radius = 1.0
    return base_pos + base_forward * base_radius


def _find_camera_element(
    camera_map: dict[str, ET.Element],
    candidate_names: Iterable[str],
) -> Optional[ET.Element]:
    """Finds the first existing camera element among candidate names."""

    for name in candidate_names:
        camera_elem = camera_map.get(name)
        if camera_elem is not None:
            return camera_elem
    return None


def _build_fixed_camera_spec(
    name: str,
    pos: np.ndarray,
    target_pos: np.ndarray,
    camera_fovy: Optional[str] = None,
) -> CameraSpec:
    """Builds a fixed camera spec that looks at ``target_pos``."""

    return CameraSpec(
        name=name,
        pos=np.asarray(pos, dtype=np.float64),
        quat=_lookat_quat_wxyz(np.asarray(pos, dtype=np.float64), target_pos),
        fovy=camera_fovy,
    )


def _generate_operation_camera_specs(
    root: ET.Element,
    config: OperationCameraConfig,
) -> list[CameraSpec]:
    """Generates the fixed operation cameras around the inferred scene center."""

    def build_back_corner_position(diagonal_rel: np.ndarray) -> np.ndarray:
        """Builds one back-corner camera position."""

        diagonal_pos = center + np.asarray(diagonal_rel, dtype=np.float64)
        diagonal_xy = diagonal_pos[:2] - center[:2]
        diagonal_xy_norm = float(np.linalg.norm(diagonal_xy))
        if diagonal_xy_norm <= 1e-8:
            diagonal_xy = np.array([1.0, 0.0], dtype=np.float64)
            diagonal_xy_norm = 1.0

        diagonal_dir = diagonal_xy / diagonal_xy_norm
        diagonal_pos[:2] = center[:2] + diagonal_dir * back_horizontal_radius
        diagonal_pos[:2] -= diagonal_dir * (0.95 * horizontal_radius)
        diagonal_pos[2] = center[2] + 0.5 * (back_rel[2] + diagonal_rel[2])
        diagonal_pos[2] += 0.42 * horizontal_radius
        return diagonal_pos

    def shift_lookat_lateral(
        camera_pos: np.ndarray,
        target_pos: np.ndarray,
        distance: float,
        side: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Shifts a look-at camera pair sideways while preserving view direction."""

        forward = _normalize(
            np.asarray(target_pos, dtype=np.float64)
            - np.asarray(camera_pos, dtype=np.float64)
        )
        up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        left_dir = np.cross(up_hint, forward)
        if np.linalg.norm(left_dir) <= 1e-8:
            up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            left_dir = np.cross(up_hint, forward)
        left_dir = _normalize(left_dir)
        if side == "right":
            left_dir = -left_dir
        shift = left_dir * float(distance)
        return (
            np.asarray(camera_pos, dtype=np.float64) + shift,
            np.asarray(target_pos, dtype=np.float64) + shift,
        )

    def shift_lookat_forward(
        camera_pos: np.ndarray,
        target_pos: np.ndarray,
        distance: float,
        direction: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Shifts a look-at camera pair forward or backward."""

        forward = _normalize(
            np.asarray(target_pos, dtype=np.float64)
            - np.asarray(camera_pos, dtype=np.float64)
        )
        if direction == "backward":
            forward = -forward
        shift = forward * float(distance)
        return (
            np.asarray(camera_pos, dtype=np.float64) + shift,
            np.asarray(target_pos, dtype=np.float64) + shift,
        )

    camera_map = {
        camera.get("name"): camera
        for camera in root.iter("camera")
        if camera.get("name") is not None
    }

    center_camera = _find_camera_element(
        camera_map,
        _dedupe_keep_order(
            [
                "frontview",
                config.base_camera_name,
                "agentview",
                "birdview",
                "sideview",
            ]
        ),
    )
    if center_camera is None:
        raise ValueError(
            "Cannot build operation cameras because no reference camera was found in XML"
        )

    center_pos, center_quat, _ = _camera_pose_from_element(center_camera)
    center = _camera_center_from_pose(center_pos, center_quat)

    fovy_camera = _find_camera_element(
        camera_map,
        _dedupe_keep_order(
            [
                "frontview",
                "birdview",
                "sideview",
                config.base_camera_name,
                "agentview",
            ]
        ),
    )
    camera_fovy = fovy_camera.get("fovy") if fovy_camera is not None else None

    front_camera = _find_camera_element(
        camera_map,
        _dedupe_keep_order(
            [
                "frontview",
                config.base_camera_name,
                "agentview",
            ]
        ),
    )
    if front_camera is not None:
        front_pos, _, _ = _camera_pose_from_element(front_camera)
        front_rel = front_pos - center
    else:
        front_rel = np.array([1.0, 0.0, 1.2], dtype=np.float64)

    if np.linalg.norm(front_rel[:2]) <= 1e-8:
        front_rel = np.array(
            [max(float(np.linalg.norm(front_rel)), 0.8), 0.0, max(front_rel[2], 1.0)],
            dtype=np.float64,
        )
    horizontal_radius = max(np.linalg.norm(front_rel[:2]), 0.8)
    front_dir = _normalize(np.array([front_rel[0], front_rel[1], 0.0], dtype=np.float64))
    back_dir = -front_dir

    side_camera = _find_camera_element(camera_map, ["sideview"])
    # Keep the evaluation repo self-contained while matching the multiview dataset
    # geometry: LIBERO's legacy `sideview` acts as the logical right-view reference.
    if side_camera is not None:
        side_pos, _, _ = _camera_pose_from_element(side_camera)
        right_rel = side_pos - center
    else:
        right_rel = _rotate_xy(front_rel, 90.0)

    top_camera = _find_camera_element(camera_map, ["birdview"])
    if top_camera is not None:
        top_pos, _, _ = _camera_pose_from_element(top_camera)
        top_rel = top_pos - center
    else:
        horizontal_radius = max(
            np.linalg.norm(front_rel[:2]),
            np.linalg.norm(right_rel[:2]),
            0.8,
        )
        top_rel = np.array(
            [
                -0.2 * horizontal_radius,
                0.0,
                max(abs(front_rel[2]) + horizontal_radius * 1.1, 1.8),
            ],
            dtype=np.float64,
        )

    left_rel = np.array([right_rel[0], -right_rel[1], right_rel[2]], dtype=np.float64)
    if np.linalg.norm(left_rel[:2]) <= 1e-8:
        left_rel = _rotate_xy(front_rel, -90.0)

    back_rel = _rotate_xy(front_rel, 180.0)
    back_horizontal_radius = max(np.linalg.norm(back_rel[:2]), 0.8)
    left_back_rel = 0.5 * (back_rel + left_rel)
    right_back_rel = 0.5 * (back_rel + right_rel)

    side_target = center + front_dir * (0.32 * horizontal_radius)
    side_target[2] -= 0.08 * horizontal_radius
    top_pos = center + top_rel + front_dir * (0.36 * horizontal_radius)
    top_pos[2] -= 0.30 * horizontal_radius
    left_pos = center + left_rel
    left_pos[2] += 0.26 * horizontal_radius
    right_pos = center + right_rel
    right_pos[2] += 0.26 * horizontal_radius
    back_pos = center + back_rel - back_dir * (0.95 * horizontal_radius)
    back_pos[2] += 0.42 * horizontal_radius
    left_back_pos = build_back_corner_position(left_back_rel)
    right_back_pos = build_back_corner_position(right_back_rel)
    left_back_pos[2] -= 0.50
    right_back_pos[2] -= 0.50
    top_target = center.copy()
    top_target[2] -= 0.60 * horizontal_radius
    top_pos = top_pos + _normalize(top_target - top_pos) * (0.22 * horizontal_radius)
    left_target = _pitch_target_up(left_pos, side_target, 0.0)
    right_target = _pitch_target_up(right_pos, side_target, 0.0)
    left_pos = _advance_along_view(left_pos, left_target, 0.18 * horizontal_radius)
    right_pos = _advance_along_view(right_pos, right_target, 0.18 * horizontal_radius)
    back_target = center.copy()
    back_target[2] += 0.40 * horizontal_radius
    back_target = _pitch_target_up(back_pos, back_target, 15.0)
    left_back_target = center.copy()
    left_back_target[2] += 0.40 * horizontal_radius
    left_back_target = _pitch_target_up(left_back_pos, left_back_target, -50.0)
    right_back_target = center.copy()
    right_back_target[2] += 0.40 * horizontal_radius
    right_back_target = _pitch_target_up(right_back_pos, right_back_target, -50.0)
    left_back_pos, left_back_target = shift_lookat_lateral(
        left_back_pos,
        left_back_target,
        0.60,
        side="right",
    )
    left_back_pos, left_back_target = shift_lookat_forward(
        left_back_pos,
        left_back_target,
        0.30,
        direction="backward",
    )
    left_back_pos, left_back_target = shift_lookat_lateral(
        left_back_pos,
        left_back_target,
        0.15,
        side="left",
    )
    right_back_pos, right_back_target = shift_lookat_lateral(
        right_back_pos,
        right_back_target,
        0.60,
        side="left",
    )
    right_back_pos, right_back_target = shift_lookat_forward(
        right_back_pos,
        right_back_target,
        0.30,
        direction="backward",
    )
    right_back_pos, right_back_target = shift_lookat_lateral(
        right_back_pos,
        right_back_target,
        0.15,
        side="right",
    )

    return [
        _build_fixed_camera_spec(
            name=config.camera_names["top"],
            pos=top_pos,
            target_pos=top_target,
            camera_fovy=camera_fovy,
        ),
        _build_fixed_camera_spec(
            name=config.camera_names["left"],
            pos=left_pos,
            target_pos=left_target,
            camera_fovy=camera_fovy,
        ),
        _build_fixed_camera_spec(
            name=config.camera_names["right"],
            pos=right_pos,
            target_pos=right_target,
            camera_fovy=camera_fovy,
        ),
        _build_fixed_camera_spec(
            name=config.camera_names["back"],
            pos=back_pos,
            target_pos=back_target,
            camera_fovy=camera_fovy,
        ),
        _build_fixed_camera_spec(
            name=config.camera_names["left_back"],
            pos=left_back_pos,
            target_pos=left_back_target,
            camera_fovy=camera_fovy,
        ),
        _build_fixed_camera_spec(
            name=config.camera_names["right_back"],
            pos=right_back_pos,
            target_pos=right_back_target,
            camera_fovy=camera_fovy,
        ),
    ]


def _inject_operation_cameras_into_arena(mujoco_arena):
    """Adds the fixed multiview operation cameras to a LIBERO arena."""
    specs = _generate_operation_camera_specs(mujoco_arena.root, OperationCameraConfig())
    for spec in specs:
        camera_attribs = {"mode": spec.mode}
        if spec.fovy is not None:
            camera_attribs["fovy"] = spec.fovy
        mujoco_arena.set_camera(
            camera_name=spec.name,
            pos=spec.pos,
            quat=spec.quat,
            camera_attribs=camera_attribs,
        )


def _wrap_setup_camera_method(setup_camera):
    """Wraps a LIBERO camera setup method to append the multiview operation cameras."""
    if getattr(setup_camera, "_openvla_multiview_wrapped", False):
        return setup_camera

    @wraps(setup_camera)
    def wrapped(self, mujoco_arena):
        setup_camera(self, mujoco_arena)
        _inject_operation_cameras_into_arena(mujoco_arena)

    wrapped._openvla_multiview_wrapped = True
    return wrapped


def _install_multiview_camera_setup_hooks():
    """Installs one-time camera setup wrappers for all LIBERO task classes."""
    global _MULTIVIEW_CAMERA_SETUP_HOOKS_INSTALLED
    if _MULTIVIEW_CAMERA_SETUP_HOOKS_INSTALLED:
        return

    from libero.libero.envs.bddl_base_domain import BDDLBaseDomain, TASK_MAPPING

    for task_cls in {BDDLBaseDomain, *TASK_MAPPING.values()}:
        setup_camera = getattr(task_cls, "_setup_camera", None)
        if setup_camera is None:
            continue
        if getattr(setup_camera, "_openvla_multiview_wrapped", False):
            continue
        setattr(task_cls, "_setup_camera", _wrap_setup_camera_method(setup_camera))

    _MULTIVIEW_CAMERA_SETUP_HOOKS_INSTALLED = True


def _validate_camera_name(camera_name):
    """Validates a logical camera name used by the evaluation script."""
    if camera_name not in SUPPORTED_CAMERA_NAMES:
        raise ValueError(
            f"Unsupported camera_name `{camera_name}`. "
            f"Choose from: {', '.join(SUPPORTED_CAMERA_NAMES)}"
        )


def resolve_camera_names(camera_names: Sequence[str]) -> tuple[str, ...]:
    """Validates and normalizes an ordered sequence of logical camera names."""
    if isinstance(camera_names, str):
        normalized_camera_names = (camera_names,)
    else:
        normalized_camera_names = tuple(str(camera_name) for camera_name in camera_names)

    if not normalized_camera_names:
        raise ValueError("Expected at least one camera name in `camera_names`.")

    invalid_camera_names = [
        camera_name for camera_name in normalized_camera_names if camera_name not in SUPPORTED_CAMERA_NAMES
    ]
    if invalid_camera_names:
        raise ValueError(
            f"Unsupported camera_name values {invalid_camera_names}. "
            f"Choose from: {', '.join(SUPPORTED_CAMERA_NAMES)}"
        )

    seen_camera_names = set()
    duplicate_camera_names = []
    for camera_name in normalized_camera_names:
        if camera_name in seen_camera_names and camera_name not in duplicate_camera_names:
            duplicate_camera_names.append(camera_name)
        seen_camera_names.add(camera_name)

    if duplicate_camera_names:
        raise ValueError(f"Duplicate camera_name values are not allowed: {duplicate_camera_names}")

    return normalized_camera_names


def _get_camera_image(obs, camera_name):
    """Returns an RGB observation frame for the requested logical camera name."""
    _validate_camera_name(camera_name)
    obs_key = CAMERA_NAME_TO_OBS_KEY[camera_name]
    if obs_key not in obs:
        available_keys = sorted(key for key in obs if key.endswith("_image"))
        raise KeyError(
            f"Observation key `{obs_key}` not found for camera `{camera_name}`. "
            f"Available image keys: {available_keys}"
        )
    return correct_libero_image_orientation(obs[obs_key])


def _maybe_resize_frame(frame, height, width):
    """Resizes a rollout frame to the requested shape if needed."""
    if frame.shape[0] == height and frame.shape[1] == width:
        return frame
    return np.asarray(Image.fromarray(frame).resize((width, height), Image.BILINEAR))


def _draw_frame_label(draw, x_offset, frame_width, label):
    """Renders a compact label onto a composed rollout frame."""
    text = str(label)
    text_left, text_top, text_right, text_bottom = draw.textbbox((0, 0), text)
    padding = 6
    left = x_offset + 8
    top = 8
    max_right = x_offset + frame_width - 8
    if max_right <= left:
        return
    right = min(left + (text_right - text_left) + 2 * padding, max_right)
    if right <= left:
        return
    bottom = top + (text_bottom - text_top) + 2 * padding
    draw.rectangle((left, top, right, bottom), fill=(0, 0, 0))
    draw.text((left + padding, top + padding), text, fill=(255, 255, 255))


def compose_rollout_frame(left_frame, right_frame, left_label, right_label):
    """Composes a side-by-side rollout frame with short labels."""
    left_frame = np.asarray(left_frame, dtype=np.uint8)
    right_frame = np.asarray(right_frame, dtype=np.uint8)
    target_height, target_width = left_frame.shape[:2]
    right_frame = _maybe_resize_frame(right_frame, target_height, target_width)
    canvas = np.concatenate([left_frame, right_frame], axis=1)
    labeled = Image.fromarray(canvas)
    draw = ImageDraw.Draw(labeled)
    _draw_frame_label(draw, 0, target_width, left_label)
    _draw_frame_label(draw, target_width, target_width, right_label)
    return np.asarray(labeled, dtype=np.uint8)


def get_libero_env(
    task,
    model_family,
    resolution=256,
    camera_name="agentview",
    camera_names: Optional[Sequence[str]] = None,
):
    """Initializes and returns the LIBERO environment, along with the task description."""
    resolved_camera_names = (
        resolve_camera_names(camera_names) if camera_names is not None else resolve_camera_names((camera_name,))
    )
    if any(camera_name != "agentview" for camera_name in resolved_camera_names):
        _install_multiview_camera_setup_hooks()

    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_camera_names = _dedupe_keep_order(
        [
            CAMERA_NAME_TO_ENV_NAME["agentview"],
            FRONT_CAMERA_NAME,
            "robot0_eye_in_hand",
            *[CAMERA_NAME_TO_ENV_NAME[camera_name] for camera_name in resolved_camera_names],
        ]
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_names": env_camera_names,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def correct_libero_image_orientation(img):
    """Converts LIBERO offscreen renders to a top-left image convention."""
    return np.flip(img, axis=0).copy()


def get_libero_image(obs, resize_size, camera_name="agentview"):
    """Extracts the model input image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = _get_camera_image(obs, camera_name)
    img = resize_image(img, resize_size)
    return img


def get_libero_images(obs, resize_size, camera_names: Sequence[str] = DEFAULT_CAMERA_NAMES):
    """Extracts an ordered sequence of model input images from observations."""
    resolved_camera_names = resolve_camera_names(camera_names)
    return np.stack(
        [get_libero_image(obs, resize_size, camera_name=camera_name) for camera_name in resolved_camera_names],
        axis=0,
    )


def get_libero_rollout_name(camera_names: Sequence[str]) -> str:
    """Returns a short rollout label derived from the evaluation camera selection."""
    resolved_camera_names = resolve_camera_names(camera_names)
    if len(resolved_camera_names) == 1 and resolved_camera_names[0] != "agentview":
        return f"{resolved_camera_names[0]}+agentview"
    return "agentview"


def get_libero_rollout_frame(obs, camera_names: Sequence[str]):
    """Builds the rollout frame for the current evaluation mode."""
    resolved_camera_names = resolve_camera_names(camera_names)
    agentview_frame = _get_camera_image(obs, "agentview")
    if len(resolved_camera_names) == 1 and resolved_camera_names[0] != "agentview":
        current_view = _get_camera_image(obs, resolved_camera_names[0])
        return compose_rollout_frame(current_view, agentview_frame, resolved_camera_names[0], "agentview")
    return agentview_frame


def _slugify_path_component(value, max_length=80):
    """Converts text into a filesystem-friendly path component."""
    normalized = value.strip().lower().replace("\n", " ")
    normalized = re.sub(r"\s+", "_", normalized)
    normalized = re.sub(r"[^a-z0-9._-]", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("._")
    if not normalized:
        return "task"
    return normalized[:max_length]


def save_rollout_video(
    rollout_images,
    idx,
    success,
    task_description,
    camera_name="agentview",
    task_name=None,
    task_suite_name=None,
    rollout_root_dir=None,
    log_file=None,
):
    """Saves an MP4 replay of an episode."""
    if rollout_root_dir is None:
        rollout_dir = f"./rollouts/{DATE}"
    else:
        suite_dir = _slugify_path_component(task_suite_name or "libero")
        task_dir = _slugify_path_component(task_name or task_description)
        rollout_dir = os.path.join(os.path.expanduser(rollout_root_dir), suite_dir, task_dir)

    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = _slugify_path_component(task_description, max_length=50)
    processed_camera_name = _slugify_path_component(camera_name, max_length=20)
    mp4_path = os.path.join(
        rollout_dir,
        f"{DATE_TIME}--episode={idx}--success={success}--camera={processed_camera_name}"
        f"--task={processed_task_description}.mp4",
    )
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
