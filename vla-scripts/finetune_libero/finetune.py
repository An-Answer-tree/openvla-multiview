"""
finetune_libero.py

Parameter-efficient fine-tuning for multiview LIBERO HDF5 datasets, keeping the official OpenVLA fine-tuning loop
structure and replacing only the dataset loader.
"""

import os
import random
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import draccus
import numpy as np
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.optimization import get_cosine_schedule_with_warmup

import wandb
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction, get_num_visual_tokens
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from camera_views import DEFAULT_CAMERA_NAMES, get_camera_views, resolve_camera_names
from dataset import LiberoMultiviewDataset

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


COSINE_WARMUP_SCHEDULER_TYPE = "linear-warmup+cosine-decay"


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                                           # Path to OpenVLA model

    # Directory Paths
    data_root_dir: Path = Path("/opt/public/liutong/LIBERO-datasets-multiview-cameraINFO-double-backsideview") # Path to multiview LIBERO root
    dataset_name: str = "libero_spatial"                                            # LIBERO benchmark directory name
    camera_names: List[str] = field(default_factory=lambda: list(DEFAULT_CAMERA_NAMES))  # Ordered camera names to include per transition
    run_root_dir: Path = Path("runs")                                               # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")                                     # Temporary directory for LoRA weights before fusing

    # Fine-tuning Parameters
    batch_size: int = 1                                                            # Fine-tuning batch size
    max_steps: int = 100_000                                                        # Max number of fine-tuning steps
    save_steps: int = 5000                                                          # Interval for checkpoint saving
    learning_rate: float = 5e-4                                                     # Fine-tuning learning rate
    use_lr_scheduler: bool = False                                                  # Whether to enable LR scheduling
    lr_scheduler_type: str = COSINE_WARMUP_SCHEDULER_TYPE                           # LR scheduler type when enabled
    warmup_ratio: float = 0.03                                                      # Fraction of training steps used for warmup
    grad_accumulation_steps: int = 16                                                # Gradient accumulation steps
    image_aug: bool = True                                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                                              # Dataloader shuffle buffer size
    seed: int = 7                                                                   # Global experiment seed
    global_shuffle_across_ranks: bool = True                                        # Shard one global shuffle stream across ranks
    save_latest_checkpoint_only: bool = False                                       # Whether to keep only the latest checkpoint

    # LoRA Arguments
    use_lora: bool = True                                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning

    # Tracking Parameters
    wandb_project: str = "openvla-7view"                                                  # Name of W&B project to log to
    wandb_entity: str = "szliutong-wuhan-university"                                # Name of entity to log under
    run_id_note: Optional[str] = None                                               # Extra note for logging, Weights & Biases
    # fmt: on


def _build_lr_scheduler(
    optimizer: AdamW,
    cfg: FinetuneConfig,
) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
    """Builds the optional learning-rate scheduler for fine-tuning."""
    if not cfg.use_lr_scheduler:
        return None

    if cfg.lr_scheduler_type != COSINE_WARMUP_SCHEDULER_TYPE:
        raise ValueError(
            f"Unsupported `lr_scheduler_type={cfg.lr_scheduler_type}`. "
            f"Expected `{COSINE_WARMUP_SCHEDULER_TYPE}`."
        )

    if not 0.0 <= cfg.warmup_ratio <= 1.0:
        raise ValueError(f"Expected `warmup_ratio` in [0, 1], got {cfg.warmup_ratio}.")

    num_warmup_steps = int(cfg.max_steps * cfg.warmup_ratio)
    return get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=cfg.max_steps,
    )


def _get_learning_rate(
    optimizer: AdamW,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
) -> float:
    """Returns the current learning rate for logging."""
    if lr_scheduler is not None:
        return float(lr_scheduler.get_last_lr()[0])
    return float(optimizer.param_groups[0]["lr"])


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    camera_names = resolve_camera_names(cfg.camera_names)
    camera_views = get_camera_views(camera_names)

    print(
        f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on LIBERO benchmark `{cfg.dataset_name}` "
        f"with cameras `{list(camera_names)}` -> `{list(camera_views)}`"
    )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+cam-{'+'.join(camera_names)}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
        f"+seed-{cfg.seed}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.use_lr_scheduler:
        exp_id += f"+sched-{cfg.lr_scheduler_type}+warmup-{cfg.warmup_ratio}"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Use the locally registered OpenVLA classes so multiview patches in this repo
    # are applied instead of the cached Hub implementation.
    processor = AutoProcessor.from_pretrained(cfg.vla_path)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
    )

    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    use_ddp = distributed_state.num_processes > 1
    if use_ddp:
        vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

    vla_module = vla.module if use_ddp else vla

    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    lr_scheduler = _build_lr_scheduler(optimizer, cfg)

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
    vla_dataset = LiberoMultiviewDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        camera_views=camera_views,
        train=True,
        image_aug=cfg.image_aug,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        seed=cfg.seed,
        global_shuffle_across_ranks=cfg.global_shuffle_across_ranks,
    )

    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )

    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()
        completed_steps = 0
        for batch_idx, batch in enumerate(dataloader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device_id),
                    attention_mask=batch["attention_mask"].to(device_id),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
                    labels=batch["labels"],
                )
                loss = output.loss

            normalized_loss = loss / cfg.grad_accumulation_steps
            normalized_loss.backward()

            num_visual_tokens = get_num_visual_tokens(batch["pixel_values"], vla_module.vision_backbone.num_patches)
            action_logits = output.logits[:, num_visual_tokens : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx

            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())

            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)
            smoothened_l1_loss = sum(recent_l1_losses) / len(recent_l1_losses)

            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
                progress.update()
                current_learning_rate = _get_learning_rate(optimizer, lr_scheduler)

                if distributed_state.is_main_process and completed_steps % 10 == 0:
                    wandb.log(
                        {
                            "train_loss": smoothened_loss,
                            "action_accuracy": smoothened_action_accuracy,
                            "l1_loss": smoothened_l1_loss,
                            "learning_rate": current_learning_rate,
                        },
                        step=completed_steps,
                    )

                if completed_steps > 0 and completed_steps % cfg.save_steps == 0:
                    if distributed_state.is_main_process:
                        print(f"Saving Model Checkpoint for Step {completed_steps}")
                        save_dir = adapter_dir if cfg.use_lora else run_dir
                        processor.save_pretrained(run_dir)
                        vla_module.save_pretrained(save_dir)

                    if use_ddp:
                        dist.barrier()

                    if cfg.use_lora:
                        base_vla = AutoModelForVision2Seq.from_pretrained(
                            cfg.vla_path,
                            torch_dtype=torch.bfloat16,
                            low_cpu_mem_usage=True,
                        )
                        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                        merged_vla = merged_vla.merge_and_unload()
                        if distributed_state.is_main_process:
                            if cfg.save_latest_checkpoint_only:
                                merged_vla.save_pretrained(run_dir)
                                print(f"Saved Model Checkpoint for Step {completed_steps} at: {run_dir}")
                            else:
                                checkpoint_dir = Path(str(run_dir) + f"--{completed_steps}_chkpt")
                                os.makedirs(checkpoint_dir, exist_ok=True)
                                save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)
                                processor.save_pretrained(checkpoint_dir)
                                merged_vla.save_pretrained(checkpoint_dir)
                                print(f"Saved Model Checkpoint for Step {completed_steps} at: {checkpoint_dir}")

                    if use_ddp:
                        dist.barrier()

                if completed_steps == cfg.max_steps:
                    print(f"Max step {cfg.max_steps} reached! Stopping training...")
                    break


if __name__ == "__main__":
    finetune()
