"""Utils for evaluating policies in LIBERO simulation environments."""

import importlib
import math
import os
import re
import sys
from functools import wraps
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import imageio
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)


SUPPORTED_CAMERA_NAMES = ("agentview", "topview", "leftview", "rightview", "backview")
FRONT_CAMERA_NAME = "frontview"
FRONT_CAMERA_OBS_KEY = f"{FRONT_CAMERA_NAME}_image"
CAMERA_NAME_TO_ENV_NAME = {
    "agentview": "agentview",
    "topview": "operation_topview",
    "leftview": "operation_leftview",
    "rightview": "operation_rightview",
    "backview": "operation_backview",
}
CAMERA_NAME_TO_OBS_KEY = {
    camera_name: f"{env_camera_name}_image"
    for camera_name, env_camera_name in CAMERA_NAME_TO_ENV_NAME.items()
}

_MULTIVIEW_CAMERA_TOOLS = None
_MULTIVIEW_CAMERA_SETUP_HOOKS_INSTALLED = False


def _dedupe_keep_order(values):
    """Returns a list without duplicates while preserving the input order."""
    return list(dict.fromkeys(values))


def _get_libero_repo_root():
    """Resolves the local LIBERO repository root used for multiview helpers."""
    if os.environ.get("LIBERO_ROOT"):
        return Path(os.environ["LIBERO_ROOT"]).expanduser().resolve()
    return Path(__file__).resolve().parents[4] / "LIBERO"


def _load_multiview_camera_tools():
    """Loads the shared multiview camera generation helpers from the LIBERO repo."""
    global _MULTIVIEW_CAMERA_TOOLS
    if _MULTIVIEW_CAMERA_TOOLS is not None:
        return _MULTIVIEW_CAMERA_TOOLS

    libero_repo_root = _get_libero_repo_root()
    scripts_dir = libero_repo_root / "scripts"
    if not scripts_dir.is_dir():
        raise FileNotFoundError(
            f"Cannot find LIBERO multiview scripts under {scripts_dir}. "
            "Set LIBERO_ROOT if the repo lives elsewhere."
        )

    for import_path in (scripts_dir, libero_repo_root):
        import_path_str = str(import_path)
        if import_path_str not in sys.path:
            sys.path.insert(0, import_path_str)

    try:
        config_module = importlib.import_module("multiview_collect_demo.config")
        camera_injection_module = importlib.import_module("multiview_collect_demo.camera_injection")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import LIBERO multiview camera helpers from {scripts_dir}: {exc}"
        ) from exc

    _MULTIVIEW_CAMERA_TOOLS = (
        config_module.OperationCameraConfig,
        camera_injection_module,
    )
    return _MULTIVIEW_CAMERA_TOOLS


def _inject_operation_cameras_into_arena(mujoco_arena):
    """Adds the fixed multiview operation cameras to a LIBERO arena."""
    operation_camera_config_cls, camera_injection_module = _load_multiview_camera_tools()
    specs = camera_injection_module._generate_operation_camera_specs(
        mujoco_arena.root,
        operation_camera_config_cls(),
    )
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


def _get_front_rollout_image(obs):
    """Returns the human-facing front rollout frame with a safe fallback."""
    obs_key = FRONT_CAMERA_OBS_KEY if FRONT_CAMERA_OBS_KEY in obs else CAMERA_NAME_TO_OBS_KEY["agentview"]
    return correct_libero_image_orientation(obs[obs_key]), obs_key.replace("_image", "")


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
    right = min(left + (text_right - text_left) + 2 * padding, x_offset + frame_width - 8)
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


def get_libero_env(task, model_family, resolution=256, camera_name="agentview"):
    """Initializes and returns the LIBERO environment, along with the task description."""
    _validate_camera_name(camera_name)
    if camera_name != "agentview":
        _install_multiview_camera_setup_hooks()

    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_camera_names = _dedupe_keep_order(
        [
            CAMERA_NAME_TO_ENV_NAME["agentview"],
            FRONT_CAMERA_NAME,
            "robot0_eye_in_hand",
            CAMERA_NAME_TO_ENV_NAME[camera_name],
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


def get_libero_rollout_frame(obs, camera_name="agentview"):
    """Builds a side-by-side rollout frame for the active view and the front view."""
    current_view = _get_camera_image(obs, camera_name)
    front_view, front_label = _get_front_rollout_image(obs)
    return compose_rollout_frame(current_view, front_view, camera_name, front_label)


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
