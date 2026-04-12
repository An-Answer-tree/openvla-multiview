"""
run_libero_eval_student.py

Runs a distilled OpenVLA student model in a LIBERO simulation environment.
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb

# Add the repository root so `experiments.*` imports do not depend on the shell cwd.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
    get_libero_images,
    get_libero_rollout_frame,
    get_libero_rollout_name,
    quat2axisangle,
    resolve_camera_names,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor  # noqa: E402
from experiments.robot.openvla_student_utils import (  # noqa: E402
    get_student_vla_action,
    load_student_vla,
)
from experiments.robot.robot_utils import (  # noqa: E402
    DATE_TIME,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


MODEL_FAMILY = "openvla_student"
IMAGE_RESIZE_SIZE = 224


@dataclass
class StudentEvalConfig:
    """Configuration for distilled student evaluation on LIBERO."""

    # Model-specific parameters.
    pretrained_checkpoint: Union[str, Path] = "PATH_TO_DISTILLED_STUDENT_CHECKPOINT"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True

    # LIBERO environment-specific parameters.
    task_suite_name: str = "libero_spatial"
    camera_names: List[str] = field(default_factory=lambda: ["leftview"])
    num_steps_wait: int = 10
    num_trials_per_task: int = 50

    # Logging and reproducibility.
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    rollout_root_dir: str = "~/Desktop/openvla_libero_rollouts"
    use_wandb: bool = False
    wandb_project: str = "YOUR_WANDB_PROJECT"
    wandb_entity: str = "YOUR_WANDB_ENTITY"
    seed: int = 7


def _validate_student_eval_camera(
    camera_names: tuple[str, ...],
    student_target_camera_names: tuple[str, ...],
) -> str:
    """Validates that eval uses one legal student camera."""
    if len(camera_names) != 1:
        raise ValueError(
            "Distilled student eval currently supports exactly one camera. "
            f"Got: {list(camera_names)}"
        )

    eval_camera_name = camera_names[0]
    if eval_camera_name not in student_target_camera_names:
        raise ValueError(
            "Requested eval camera "
            f"`{eval_camera_name}` is not present in the student checkpoint. "
            f"Choose from: {list(student_target_camera_names)}"
        )

    return eval_camera_name


def _get_image_resize_size() -> int:
    """Returns the image size expected by OpenVLA student checkpoints."""
    return IMAGE_RESIZE_SIZE


@draccus.wrap()
def eval_libero_student(cfg: StudentEvalConfig) -> None:
    """Runs one distilled student policy in a LIBERO simulation environment."""
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    camera_names = resolve_camera_names(cfg.camera_names)
    rollout_name = get_libero_rollout_name(camera_names)

    set_seed_everywhere(cfg.seed)
    cfg.unnorm_key = cfg.task_suite_name

    model = load_student_vla(cfg)
    eval_camera_name = _validate_student_eval_camera(camera_names, model.student_target_camera_names)

    if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    assert cfg.unnorm_key in model.norm_stats, (
        f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"
    )

    processor = get_processor(cfg)

    camera_id = "+".join(camera_names)
    run_id = f"EVAL-{cfg.task_suite_name}-{MODEL_FAMILY}-cam-{camera_id}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")

    resolved_rollout_root_dir = os.path.expanduser(cfg.rollout_root_dir)
    print(f"Loaded model: {type(model)}")
    print(f"Logging to local log file: {local_log_filepath}")
    print(f"Rollout videos will be saved under: {resolved_rollout_root_dir}")
    print(f"Evaluation cameras: {list(camera_names)}; rollout view: `{rollout_name}`.")
    log_file.write(f"Rollout videos will be saved under: {resolved_rollout_root_dir}\n")
    log_file.write(f"Evaluation cameras: {list(camera_names)}; rollout view: `{rollout_name}`.\n")

    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    log_file.write(f"Camera names: {list(camera_names)}\n")
    log_file.write(f"Rollout view: {rollout_name}\n")

    resize_size = _get_image_resize_size()
    total_episodes, total_successes = 0, 0

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        if cfg.num_trials_per_task > len(initial_states):
            raise ValueError(
                "num_trials_per_task exceeds the available initial states for this task: "
                f"{cfg.num_trials_per_task} > {len(initial_states)}"
            )

        env, task_description = get_libero_env(
            task,
            MODEL_FAMILY,
            resolution=256,
            camera_names=camera_names,
        )

        try:
            task_episodes, task_successes = 0, 0
            for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
                print(f"\nTask: {task_description}")
                log_file.write(f"\nTask: {task_description}\n")

                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])

                t = 0
                done = False
                replay_images = []
                if cfg.task_suite_name == "libero_spatial":
                    max_steps = 220
                elif cfg.task_suite_name == "libero_object":
                    max_steps = 280
                elif cfg.task_suite_name == "libero_goal":
                    max_steps = 300
                elif cfg.task_suite_name == "libero_10":
                    max_steps = 520
                elif cfg.task_suite_name == "libero_90":
                    max_steps = 400
                else:
                    raise ValueError(f"Unsupported task suite `{cfg.task_suite_name}`.")

                print(f"Starting episode {task_episodes + 1}...")
                log_file.write(f"Starting episode {task_episodes + 1}...\n")
                while t < max_steps + cfg.num_steps_wait:
                    try:
                        if t < cfg.num_steps_wait:
                            obs, reward, done, info = env.step(get_libero_dummy_action(MODEL_FAMILY))
                            t += 1
                            continue

                        imgs = get_libero_images(obs, resize_size, camera_names)
                        rollout_img = get_libero_rollout_frame(obs, camera_names)
                        replay_images.append(rollout_img)

                        observation = {
                            "full_image": imgs[0],
                            "full_images": imgs,
                            "state": np.concatenate(
                                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                            ),
                        }

                        action = get_student_vla_action(
                            model=model,
                            processor=processor,
                            base_vla_name=str(cfg.pretrained_checkpoint),
                            obs=observation,
                            task_label=task_description,
                            unnorm_key=cfg.unnorm_key,
                            student_camera_name=eval_camera_name,
                            center_crop=cfg.center_crop,
                        )

                        action = normalize_gripper_action(action, binarize=True)
                        action = invert_gripper_action(action)

                        obs, reward, done, info = env.step(action.tolist())
                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                        t += 1

                    except Exception as exc:
                        print(f"Caught exception: {exc}")
                        log_file.write(f"Caught exception: {exc}\n")
                        break

                task_episodes += 1
                total_episodes += 1

                save_rollout_video(
                    replay_images,
                    total_episodes,
                    success=done,
                    task_description=task_description,
                    camera_name=rollout_name,
                    task_name=task.name,
                    task_suite_name=cfg.task_suite_name,
                    rollout_root_dir=cfg.rollout_root_dir,
                    log_file=log_file,
                )

                print(f"Success: {done}")
                print(f"# episodes completed so far: {total_episodes}")
                print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
                log_file.write(f"Success: {done}\n")
                log_file.write(f"# episodes completed so far: {total_episodes}\n")
                log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
                log_file.flush()

            print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
            print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
            log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
            log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
            log_file.flush()
            if cfg.use_wandb:
                wandb.log(
                    {
                        f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                        f"num_episodes/{task_description}": task_episodes,
                    }
                )
        finally:
            env.close()

    log_file.close()

    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero_student()
