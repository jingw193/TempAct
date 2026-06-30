# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================

from typing import Any, List
from collections import defaultdict
import os
import datetime
from concurrent import futures
import time
import json
from absl import app, flags
import logging
from diffusers import StableDiffusion3Pipeline
import numpy as np
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
from torch.utils.data.distributed import DistributedSampler
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper
from ml_collections import config_flags
from self_forcing.causal_pipeline import CausalInferencePipeline
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
from einops import rearrange
from torchvision.io import write_video
from accelerate import Accelerator
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from self_forcing.compute_log_prob_self_forcing import compute_log_prob
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = True
torch._inductor.config.debug = True

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


# ------------------------------------------------------------------ #
#  工具函数
# ------------------------------------------------------------------ #

def set_seed(seed: int, rank: int = 0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def collate_keep_list(values: List[Any]) -> Any:
    """
    Collate rule:
      - Tensor  -> torch.cat(dim=0)
      - Dict    -> recursively collate
      - List    -> keep as List (do NOT merge)
    """
    v0 = values[0]

    # -------- Tensor --------
    if torch.is_tensor(v0):
        return torch.cat(values, dim=0)

    # -------- Dict --------
    if isinstance(v0, dict):
        return {
            k: collate_keep_list([v[k] for v in values])
            for k in v0.keys()
        }

    # -------- List --------
    if isinstance(v0, list):
        # Do NOT merge list, keep per-sample structure
        return values

    # -------- Other types --------
    return values


def gather_tensor_to_all(tensor, accelerator):
    """通过 Accelerate 在所有进程间 all-gather 张量，返回 CPU 张量。"""
    gathered = accelerator.gather(tensor)
    return gathered.cpu()


def return_decay(step, decay_type):
    if decay_type == 0:
        flat = 0
        uprate = 0.0
        uphold = 0.0
    elif decay_type == 1:
        flat = 0
        uprate = 0.001
        uphold = 0.5
    elif decay_type == 2:
        flat = 75
        uprate = 0.0075
        uphold = 0.999
    else:
        assert False

    if step < flat:
        return 0.0
    else:
        decay = (step - flat) * uprate
        return min(decay, uphold)


def calculate_zero_std_ratio(prompts, gathered_rewards):
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(prompt_array, return_inverse=True, return_counts=True)
    grouped_rewards = gathered_rewards["avg"].mean(-1)[np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)

    diffusion_group_std = gathered_rewards["avg"].std(-1)
    diffusion_zero_std_ratio = (diffusion_group_std < 1e-8).mean()
    return zero_std_ratio, prompt_std_devs.mean(), diffusion_zero_std_ratio, diffusion_group_std.mean()


# ------------------------------------------------------------------ #
#  数据集 & 采样器
# ------------------------------------------------------------------ #

class TextPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}.txt")
        with open(self.file_path, "r") as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class TextPromptVideoDataset(Dataset):
    def __init__(self, dataset, split="train", debug=False):
        if split == "train":
            if dataset == "temporal_order":
                self.file_paths = [
                    "/path/to/TempAct/dataset/temporal_order/general_order_temporal_prompts_5k.csv",
                ]
            else:
                self.file_paths = []
        else:
            if dataset == "temporal_order":
                self.file_paths = [
                    "/path/to/TempAct/dataset/temporal_order/general_order_temporal_prompts_eval_128.csv",
                ]
            else:
                self.file_paths = []


        self.prompts = []

        df_list = [pd.read_csv(p) for p in self.file_paths]
        df = pd.concat(df_list, ignore_index=True)

        for _, row in tqdm(df.iterrows(), total=len(df)):
            self.prompts.append(row["prompt"])

        if split == "train" and dataset != "temporal_order":
            pickscore_data_file_path = "/path/to/TempAct/dataset/pickscore/train.txt"
            with open(pickscore_data_file_path, "r") as f:
                pick_score_prompts = [line.strip() for line in f.readlines()]
            self.new_prompts = self.prompts + pick_score_prompts
        else:
            self.new_prompts = self.prompts

    def __len__(self):
        return len(self.new_prompts)

    def __getitem__(self, idx):
        return {"prompt": self.new_prompts[idx], "metadata": {"eval_sample_frames": 6}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}_metadata.jsonl")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item["prompt"] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.total_samples = self.num_replicas * self.batch_size
        assert (
            self.total_samples % self.k == 0
        ), f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]

            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt, max_sequence_length)
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


# ------------------------------------------------------------------ #
#  eval_fn  —  用 accelerator 替代 rank / world_size / mixed_precision_dtype
# ------------------------------------------------------------------ #

def eval_fn(
    pipeline,
    test_dataloader,
    unwrapped_transformer,
    unwrapped_llm,
    llm_policy_tokenizer,
    tokenizers,
    config,
    accelerator,
    global_step,
    reward_fn,
    executor,
    ema,
    transformer_trainable_parameters,
):
    device = accelerator.device
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    if config.train.ema and ema is not None:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    all_rewards = defaultdict(list)

    test_sampler = (
        DistributedSampler(test_dataloader.dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )
    eval_loader = DataLoader(
        test_dataloader.dataset,
        batch_size=config.sample.test_batch_size,
        sampler=test_sampler,
        collate_fn=test_dataloader.collate_fn,
        num_workers=test_dataloader.num_workers,
    )

    for test_batch in tqdm(
        eval_loader,
        desc="Eval: ",
        disable=not accelerator.is_main_process,
        position=0,
    ):
        prompts, prompt_metadata = test_batch[0], test_batch[1]
        
        prompt_ids, _ = tokenizers[0](prompts, return_mask=True, add_special_tokens=True)
        prompt_ids = prompt_ids.to(device)

        sampled_noise = torch.randn(
            [len(prompts), 27, 16,
                config.train_height // 8,
                config.train_width // 8], device=device, dtype=torch.bfloat16
        )

        with torch.no_grad():
            collate_data = pipeline.flowgrpo_inference_withllmpolicy_Diffusionmix_sample(
                noise=sampled_noise,
                text_prompts=prompts,
                policy_model=None,
                llm_semantic_policy=unwrapped_llm,
                llm_policy_tokenizer=llm_policy_tokenizer,
                low_memory=False,
                return_log_prob=True,
                gap_frame=9,
                inner_diffusion_group=1,
                inner_diffusion_frame_step=9,
                accelerator=accelerator
            )

        for prompt_metadata_item in prompt_metadata:
            prompt_metadata_item["train_frames"] = 9
            prompt_metadata_item["inner_group"] = 1
            prompt_metadata_item["gap_frame"] = 9

        prompt_for_rewards = collate_data["prompt_for_reward"]
        prompt_for_llm_rewards = collate_data["prompt_for_llm_rewards"]
        for prompt_for_rewards_index, (prompt_for_reward, prompt_for_llm_reward) in enumerate(zip(prompt_for_rewards, prompt_for_llm_rewards)):
            prompt_metadata[prompt_for_rewards_index]["decomposed_outputs"] = prompt_for_reward
            prompt_metadata[prompt_for_rewards_index]["decomposed_llm_outputs"] = prompt_for_llm_reward

        if collate_data["video"].dim() == 6:
            collate_data["video"] = collate_data["video"].reshape(-1, *collate_data["video"].shape[2:])
            rewards_future = executor.submit(
                reward_fn, 
                collate_data["video"], 
                [x for x in prompts for _ in range(1)], 
                prompt_metadata * 1, 
                only_strict=True
            )
        time.sleep(0)
        rewards, reward_metadata = rewards_future.result()
        rewards = {k: torch.as_tensor(v, device=device).float() for k, v in rewards.items()}

        # To do
        reward_keys = list(rewards.keys())
        for k in reward_keys:
            v = rewards[k]
            gathered_value = gather_tensor_to_all(v, accelerator)
            all_rewards[k].append(gathered_value.numpy())

        format_gathered_value = gather_tensor_to_all(collate_data["format_reward"], accelerator).float()
        all_rewards["format_reward"].append(format_gathered_value.numpy())

    if accelerator.is_main_process:
        videos = rearrange(collate_data["video"], 'b t c h w -> b t h w c')
        final_rewards = {key: np.concatenate(value_list) for key, value_list in all_rewards.items()}

        videos_to_log = videos.cpu()
        prompts_to_log = prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples_to_log = min(15, len(prompts_to_log))
            for idx in range(num_samples_to_log):
                video = videos_to_log[idx] * 255.0
                write_video(os.path.join(tmpdir, f"{idx}.mp4"), video, fps=16)

            sampled_prompts_log = [prompts_to_log[i] for i in range(num_samples_to_log)]
            sampled_rewards_log = [{k: final_rewards[k][i] for k in final_rewards} for i in range(num_samples_to_log)]

            wandb.log(
                {
                    "eval_videos": [
                        wandb.Video(
                            os.path.join(tmpdir, f"{idx}.mp4"),
                            caption=f"{prompt:.100} | avg: {reward}",
                            format="mp4",
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts_log, sampled_rewards_log))
                    ],
                    **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in final_rewards.items()},
                },
                step=global_step,
            )

    if config.train.ema and ema is not None:
        ema.copy_temp_to(transformer_trainable_parameters)

    accelerator.wait_for_everyone()


# ------------------------------------------------------------------ #
#  save_ckpt  —  用 accelerator.unwrap_model / is_main_process
# ------------------------------------------------------------------ #

def save_ckpt(
    save_dir, transformer, llm_policy, global_step, accelerator,
    ema, transformer_trainable_parameters, config, optimizer
):
    if accelerator.is_main_process:
        save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")

        if config.use_llm_lora:
            save_root_llm_lora = os.path.join(save_root, "llm_lora")
            os.makedirs(save_root_llm_lora, exist_ok=True)
            llm_model_to_save = accelerator.unwrap_model(llm_policy)
            llm_model_to_save.save_pretrained(save_root_llm_lora)

        save_root_lora = os.path.join(save_root, "lora")
        os.makedirs(save_root_lora, exist_ok=True)

        model_to_save = accelerator.unwrap_model(transformer)

        if config.train.ema and ema is not None:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

        model_to_save.save_pretrained(save_root_lora)

        print(f"Saved policy weights to {save_root}")

        torch.save(optimizer.state_dict(), os.path.join(save_root, "optimizer.pt"))

        if config.train.ema and ema is not None:
            ema.copy_temp_to(transformer_trainable_parameters)
        logger.info(f"Saved checkpoint to {save_root}")


# ------------------------------------------------------------------ #
#  main
# ------------------------------------------------------------------ #

def main(_):
    config = FLAGS.config
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # 在 Accelerator 初始化之前，手动设置当前设备
    torch.cuda.set_device(local_rank)
    # ---- 创建 Accelerator（替代手动 NCCL init + GradScaler + autocast） ----
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )

    # 从 accelerator 获取分布式信息（替代手动读取环境变量）
    rank = accelerator.process_index          # 全局 rank
    world_size = accelerator.num_processes    # 总进程数
    local_rank = accelerator.local_process_index  # 节点内 rank
    device = accelerator.device              # 当前进程对应的设备

    # 保留 mixed_precision_dtype 用于模型加载时的 dtype 指定
    mixed_precision_dtype = None
    if config.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16
    enable_amp = mixed_precision_dtype is not None

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # ---- WandB（仅主进程）----
    if accelerator.is_main_process:
        log_dir = os.path.join(config.logdir, config.run_name)
        os.makedirs(log_dir, exist_ok=True)
        if config.debug:
            wandb.init(project="debug3", name=config.run_name, config=config.to_dict(), dir=log_dir)
        else:
            # wandb.init(project="self_forcing_diffusion", name=config.run_name, config=config.to_dict(), dir=log_dir)
            wandb.init(project="self_forcing_llm_qwen3vl", name=config.run_name, config=config.to_dict(), dir=log_dir)

    logger.info(f"\n{config}")

    set_seed(config.seed, rank)

    # ---- 加载模型 ----
    pipeline = CausalInferencePipeline(config.pretrained.self_config, device=device)

    if config.self_model_path:
        state_dict = torch.load(config.self_model_path, map_location="cpu")
        pipeline.generator.load_state_dict(state_dict['generator_ema'])

    llm_policy_tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path, trust_remote_code=True)
    llm_policy_tokenizer.padding_side = "left"

    llm_policy = AutoModelForCausalLM.from_pretrained(
        config.llm_model_path,
        torch_dtype=torch.float16,   # LLM 单独使用 fp16，diffusion 保持 bf16
        trust_remote_code=True
    )

    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.generator.model.requires_grad_(not config.use_lora)
    llm_policy.requires_grad_(False)

    tokenizers = [pipeline.text_encoder.tokenizer]

    pipeline.vae.to(device, dtype=torch.float32)
    transformer = pipeline.generator.model.to(device)
    llm_policy.to(device)
    pipeline.text_encoder.to(device, dtype=mixed_precision_dtype if enable_amp else torch.float32)

    # ---- LoRA 配置（必须在 accelerator.prepare 之前完成）----
    if config.use_lora:
        target_modules = [
            "self_attn.q",
            "self_attn.k",
            "self_attn.v",
            "self_attn.o",
            "ffn.0",
            "ffn.2",
        ]
        transformer_lora_config = LoraConfig(
            r=256, lora_alpha=256, init_lora_weights="gaussian", target_modules=target_modules
        )
        if config.train.lora_path:
            transformer = PeftModel.from_pretrained(transformer, config.train.lora_path)
            transformer.set_adapter("default")
        else:
            transformer = get_peft_model(transformer, transformer_lora_config)
        transformer.print_trainable_parameters()

    if config.use_llm_lora:
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            task_type="CAUSAL_LM"
        )
        llm_lora_path = config.train.llm_lora_path
        if llm_lora_path is not None:
            print("load lora from lora_path")
            llm_policy = PeftModel.from_pretrained(llm_policy, llm_lora_path)
            llm_policy.set_adapter("default")
            llm_policy.print_trainable_parameters()
        else:
            llm_policy = get_peft_model(llm_policy, lora_config)
            llm_policy.set_adapter("default")
            llm_policy.print_trainable_parameters()
    else:
        llm_lora_path = config.train.llm_lora_path
        if llm_lora_path is not None:
            print("load lora from lora_path")
            llm_policy = PeftModel.from_pretrained(llm_policy, llm_lora_path)
            llm_policy.set_adapter("default")
            llm_policy.print_trainable_parameters()
            llm_policy.requires_grad_(False)


    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ---- 优化器 ----
    # 先收集可训练参数（prepare 之前）
    transformer.set_adapter("default")
    transformer_trainable_parameters_pre = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    if config.use_llm_lora:
        llm_policy_trainable_parameters_pre = list(filter(lambda p: p.requires_grad, llm_policy.parameters()))
    else:
        llm_policy_trainable_parameters_pre = []

    optimizer = torch.optim.AdamW(
        [
            {"params": transformer_trainable_parameters_pre, "lr": config.train.learning_rate},
            {"params": llm_policy_trainable_parameters_pre, "lr": config.train.llm_learning_rate},
        ],
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # ---- accelerator.prepare（替代 DDP 包装）----
    # DataLoader 不经过 prepare，因为 DistributedKRepeatSampler 已自行分片
    
    transformer, llm_policy, optimizer = accelerator.prepare(transformer, llm_policy, optimizer)

    # unwrap 别名：用于访问底层 PEFT 方法（set_adapter / disable_adapter / save_pretrained 等）
    unwrapped_transformer = accelerator.unwrap_model(transformer)
    unwrapped_llm = accelerator.unwrap_model(llm_policy)
    # ---- 解除 Accelerate 为 llm_policy 注入的 bf16 混合精度 hook ----
    # Accelerator 会为所有 prepare 过的模型注册 forward hook，将输入自动转为
    # mixed_precision_dtype（此处为 bf16）。LLM 需要单独保持 fp16，因此
    # 需要手动移除这些 hook，并在后续前向时用 fp16 autocast 替代。
    from accelerate.hooks import remove_hook_from_module
    remove_hook_from_module(unwrapped_llm, recurse=True)

    # prepare 后重新获取可训练参数引用
    unwrapped_transformer.set_adapter("default")
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, unwrapped_transformer.parameters()))

    llm_policy_trainable_parameters = list(filter(lambda p: p.requires_grad, unwrapped_llm.parameters()))

    # ---- 数据集 & DataLoader ----
    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, "train")
        test_dataset = TextPromptDataset(config.dataset, "test")
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, "train")
        test_dataset = GenevalPromptDataset(config.dataset, "test")
    if config.prompt_fn == "self_forcing":
        train_dataset = TextPromptVideoDataset(config.dataset, "train", debug=config.debug)
        test_dataset = TextPromptVideoDataset(config.dataset, "test", debug=config.debug)
    else:
        raise NotImplementedError("Prompt function not supported with dataset")

    # DistributedKRepeatSampler 使用 accelerator 的分布式信息
    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,
        k=config.sample.num_image_per_prompt,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=config.seed,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_sampler=train_sampler, num_workers=0,
        collate_fn=train_dataset.collate_fn, pin_memory=True
    )

    test_sampler = (
        DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1 else None
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,
        sampler=test_sampler,
        collate_fn=test_dataset.collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)
    else:
        assert False

    executor = futures.ThreadPoolExecutor(max_workers=1)

    # ---- 训练统计信息 ----
    samples_per_epoch = config.sample.train_batch_size * world_size * config.sample.num_batches_per_epoch
    total_train_batch_size = config.train.batch_size * world_size * config.train.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    reward_fn = getattr(flow_grpo.rewards, "multi_score")(device, config.reward_fn)
    eval_reward_fn = getattr(flow_grpo.rewards, "multi_score")(device, config.reward_fn)

    # ---- 断点恢复 ----
    first_epoch = 0
    global_step = 0
    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")

        lora_path = os.path.join(config.resume_from, "lora")
        if os.path.exists(lora_path):
            unwrapped_transformer.load_adapter(lora_path, adapter_name="default", is_trainable=True)
        else:
            model_ckpt_path = os.path.join(config.resume_from, "transformer_model.pt")
            if os.path.exists(model_ckpt_path):
                unwrapped_transformer.load_state_dict(torch.load(model_ckpt_path, map_location=device))

        llm_lora_path = os.path.join(config.resume_from, "llm_lora")
        if os.path.exists(llm_lora_path):
            unwrapped_llm.load_adapter(llm_lora_path, adapter_name="default", is_trainable=True)

        opt_path = os.path.join(config.resume_from, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))

        try:
            global_step = int(os.path.basename(config.resume_from).split("-")[-1])
            logger.info(f"Resumed global_step to {global_step}.")
        except ValueError:
            logger.warning(
                f"Could not parse global_step from checkpoint name: {config.resume_from}. Starting from 0."
            )
            global_step = 0

    # ---- EMA ----
    ema = None
    if config.train.ema:
        ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=1, device=device)

    num_train_timesteps = pipeline.denoising_step_list.shape[0]

    logger.info("***** Running training *****")

    train_iter = iter(train_dataloader)
    optimizer.zero_grad()

    # ================================================================ #
    #  训练主循环
    # ================================================================ #
    for epoch in range(first_epoch, config.num_epochs):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        # ============================================================ #
        #  采样阶段
        # ============================================================ #
        rng = random.Random(epoch)  # 独立 RNG
        diffusion_sample_frame = rng.choice(config.diffusion_sample_frames)

        pipeline.generator.eval()
        pipeline.text_encoder.eval()
        samples_data_list = []

        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_main_process,
            position=0,
        ):
            unwrapped_transformer.set_adapter("default")
            if hasattr(train_sampler, "set_epoch") and isinstance(train_sampler, DistributedKRepeatSampler):
                train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)

            prompts, prompt_metadata = next(train_iter)

            prompt_ids, _ = tokenizers[0](prompts, return_mask=True, add_special_tokens=True)
            prompt_ids = prompt_ids.to(device)

            sampled_noise = torch.randn(
                [len(prompts), config.sample_frames, 16,
                 config.train_height // 8,
                 config.train_width // 8], device=device, dtype=torch.bfloat16
            )

            if i == 0 and epoch % config.save_freq == 0 and accelerator.is_main_process and not config.debug:
                save_ckpt(
                    config.save_dir,
                    transformer,
                    llm_policy,
                    global_step,
                    accelerator,
                    ema,
                    transformer_trainable_parameters,
                    config,
                    optimizer,
                )

            with torch.no_grad():
                collate_data = pipeline.flowgrpo_inference_withllmpolicy_Diffusionmix_sample(
                    noise=sampled_noise,
                    text_prompts=prompts,
                    policy_model=None,
                    llm_semantic_policy=unwrapped_llm,
                    llm_policy_tokenizer=llm_policy_tokenizer,
                    low_memory=False,
                    return_log_prob=True,
                    gap_frame=config.gap_frame,
                    inner_diffusion_group=config.inner_diffusion_groupsize,
                    inner_diffusion_frame_step=diffusion_sample_frame,
                    accelerator=accelerator
                )

            timesteps = torch.tensor(config.pretrained.self_config.denoising_step_list).repeat(len(prompts), 1).to(device)
            # train_frames_ratio = diffusion_sample_frame / config.sample_frames
            for prompt_metadata_item in prompt_metadata:
                prompt_metadata_item["train_frames"] = diffusion_sample_frame
                prompt_metadata_item["inner_group"] = config.inner_diffusion_groupsize
                prompt_metadata_item["gap_frame"] = config.gap_frame

            prompt_for_rewards = collate_data["prompt_for_reward"]
            prompt_for_llm_rewards = collate_data["prompt_for_llm_rewards"]
            for prompt_for_rewards_index, (prompt_for_reward, prompt_for_llm_reward) in enumerate(zip(prompt_for_rewards, prompt_for_llm_rewards)):
                prompt_metadata[prompt_for_rewards_index]["decomposed_outputs"] = prompt_for_reward
                prompt_metadata[prompt_for_rewards_index]["decomposed_llm_outputs"] = prompt_for_llm_reward

            if collate_data["video"].dim() == 6:
                collate_data["video"] = collate_data["video"].reshape(-1, *collate_data["video"].shape[2:])
                rewards_future = executor.submit(
                    reward_fn, 
                    collate_data["video"], 
                    [x for x in prompts for _ in range(config.inner_diffusion_groupsize)],
                    [x for x in prompt_metadata for _ in range(config.inner_diffusion_groupsize)],
                    only_strict=True
                )
            else:
                rewards_future = executor.submit(reward_fn, collate_data["video"], prompts, prompt_metadata, only_strict=True)
            time.sleep(0)

            samples_data_list.append(
                {
                    "prompt_ids": prompt_ids,
                    # diffusion policy features
                    "prompt_embeds": collate_data["prompt_embedings"], # b, L, N
                    "timesteps": timesteps,
                    "latents_clean": collate_data["latent"],
                    "diffusion_log_prob": collate_data["diffusion_log_prob"], # B, G, T, 3, C, H, W
                    "diffusion_train_latents": collate_data["diffusion_train_latents"], # B, G, T, 3, C, H, W
                    "diffusion_flow_preds": collate_data["diffusion_flow_preds"], # B, G, T, 3, C, H, W
                    # llm policy features
                    "old_gen_log_probs": collate_data["old_gen_log_probs"], # B, L
                    "llm_prompt_ids": collate_data["llm_prompt_ids"], # B, L_1
                    "llm_generation_ids": collate_data["llm_generation_ids"], # B, L
                    "format_reward": collate_data["format_reward"], # B
                    "respones_json_mask": collate_data["respones_json_mask"], # B, L
                    # reward 
                    "rewards_future": rewards_future,
                }
            )

        max_respones_len = max([sample["llm_generation_ids"].shape[1] for sample in samples_data_list])
        max_input_len = max([sample["llm_prompt_ids"].shape[1] for sample in samples_data_list])

        # ============================================================ #
        #  等待奖励计算完成
        # ============================================================ #
        for sample_item in tqdm(
            samples_data_list, desc="Waiting for rewards",
            disable=not accelerator.is_main_process, position=0
        ):
            respones_pad_len = max_respones_len - sample_item["llm_generation_ids"].shape[1]
            input_pad_len = max_input_len - sample_item["llm_prompt_ids"].shape[1]
            
            sample_item["llm_generation_ids"] = torch.nn.functional.pad(
                sample_item["llm_generation_ids"],  # [B, L]
                (0, respones_pad_len),              # pad dim=1 (L)
                value=llm_policy_tokenizer.pad_token_id,
            )

            sample_item["old_gen_log_probs"] = torch.nn.functional.pad(
                sample_item["old_gen_log_probs"],  # [B, L]
                (0, respones_pad_len),              # pad dim=1 (L)
                value=0,
            )

            sample_item["respones_json_mask"] = torch.nn.functional.pad(
                sample_item["respones_json_mask"],  # [B, L]
                (0, respones_pad_len),              # pad dim=1 (L)
                value=0,
            )

            sample_item["llm_prompt_ids"] = torch.nn.functional.pad(
                sample_item["llm_prompt_ids"],  # [B, L]
                (input_pad_len, 0),              # pad dim=1 (L)
                value=llm_policy_tokenizer.pad_token_id,
            )

            rewards, reward_metadata = sample_item["rewards_future"].result()
            sample_item["rewards"] = {k: torch.as_tensor(v, device=device).float() for k, v in rewards.items()}
            sample_item["rewards"]["videopickscore_local_score"] = sample_item["rewards"]["videopickscore_local_score"].reshape(-1, config.inner_diffusion_groupsize)
            sample_item["rewards"]["qwen3vl_video_score"] = sample_item["rewards"]["qwen3vl_video_score"].reshape(-1, config.inner_diffusion_groupsize)
            # 先复制键列表，避免在迭代过程中修改字典
            reward_keys = list(sample_item["rewards"].keys())
            for k in reward_keys:
                sample_item["rewards"][k] = sample_item["rewards"][k].reshape(-1, config.inner_diffusion_groupsize)

                if k == "avg":
                    sample_item["rewards"][k] = sample_item["rewards"][k] - 0.5 * (1 - sample_item["format_reward"][:, None].float())
                    # get inner highest and lowest
                
                if k == "qwen3vl_local_score":
                    max_reward, highest_indice = sample_item["rewards"][k].max(dim=1)
                    min_reward, lowest_indice = sample_item["rewards"][k].min(dim=1)
                    sample_item["diffusion_reward_diff"] = max_reward - min_reward
                    sample_item["highest_latents"] = sample_item["diffusion_train_latents"][torch.arange(len(prompts)), highest_indice]
                    sample_item["lowest_latents"] = sample_item["diffusion_train_latents"][torch.arange(len(prompts)), lowest_indice]
                    sample_item["highest_indice"] = highest_indice
                    sample_item["lowest_indice"] = lowest_indice

                    sample_item["rewards"]["diffusion_reward_avg"] = sample_item["rewards"][k] * 0.6 + 0.4 * sample_item["rewards"]["videopickscore_local_score"]
                    sample_item["diffusion_advantages"] = sample_item["rewards"]["diffusion_reward_avg"] - sample_item["rewards"]["diffusion_reward_avg"].mean(dim=1, keepdim=True)

            sample_item["rewards"]["format_reward"] = sample_item["format_reward"].float()
            
            del sample_item["format_reward"]
            del sample_item["rewards_future"]

        collated_samples = collate_keep_list(samples_data_list)

        # ---- 记录视频（主进程）----
        if epoch % 10 == 0 and accelerator.is_main_process:
            videos = rearrange(collate_data["video"], 'b t c h w -> b t h w c')
            videos_to_log = videos.cpu()
            prompts_to_log = prompts * config.inner_diffusion_groupsize
            rewards_to_log = collated_samples["rewards"]["avg"].flatten().cpu()
            qwen3vl_local_score_rewards_to_log = collated_samples["rewards"]["qwen3vl_local_score"].flatten().cpu()

            with tempfile.TemporaryDirectory() as tmpdir:
                num_to_log = min(15, len(prompts_to_log))
                for idx in range(num_to_log):
                    video = videos_to_log[idx] * 255.0
                    write_video(os.path.join(tmpdir, f"{idx}.mp4"), video, fps=16)

                wandb.log(
                    {
                        "video": [
                            wandb.Video(
                                os.path.join(tmpdir, f"{idx}.mp4"),
                                caption=f"{prompts_to_log[idx]:.100} | avg: {rewards_to_log[idx]:.2f} | qwen3vl_local_score: {qwen3vl_local_score_rewards_to_log[idx]:.2f}",
                                format="mp4",
                            )
                            for idx in range(num_to_log)
                        ],
                    },
                    step=global_step,
                )

        # ---- gather 奖励（accelerator.gather 替代 dist.all_gather）----
        gathered_rewards_dict = {}
        for key, value_tensor in collated_samples["rewards"].items():
            if key == "diffusion_reward_avg":
                global_diffusion_reward_std = gather_tensor_to_all(value_tensor, accelerator).std()
            gathered_rewards_dict[key] = gather_tensor_to_all(value_tensor, accelerator).numpy()

        if accelerator.is_main_process:
            wandb.log(
                {
                    "epoch": epoch,
                    **{
                        f"reward_{k}": v.mean()
                        for k, v in gathered_rewards_dict.items()
                        if "_strict_accuracy" not in k and "_accuracy" not in k
                    },
                },
                step=global_step,
            )

        # ---- 优势计算 ----
        if config.per_prompt_stat_tracking:
            prompt_ids_all = gather_tensor_to_all(collated_samples["prompt_ids"], accelerator)
            prompts_all_decoded = [
                pipeline.text_encoder.tokenizer.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in prompt_ids_all.cpu().numpy()
            ]

            advantages = stat_tracker.update(prompts_all_decoded, gathered_rewards_dict["avg"].mean(-1))

            if accelerator.is_main_process:
                group_size, trained_prompt_num = stat_tracker.get_stats()
                llm_zero_std_ratio, llm_reward_std_mean, diffusion_zero_std_ratio, diffusion_reward_std_mean = calculate_zero_std_ratio(prompts_all_decoded, gathered_rewards_dict)

                wandb.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "llm_zero_std_ratio": llm_zero_std_ratio,
                        "llm_reward_std_mean": llm_reward_std_mean,
                        "diffusion_zero_std_ratio": diffusion_zero_std_ratio,
                        "diffusion_reward_std_mean": diffusion_reward_std_mean,
                    },
                    step=global_step,
                )
            stat_tracker.clear()
        else:
            avg_rewards_all = gathered_rewards_dict["avg"]
            advantages = (avg_rewards_all - avg_rewards_all.mean()) / (avg_rewards_all.std() + 1e-4)

        # 将优势分发回当前进程
        samples_per_gpu = collated_samples["prompt_embeds"].shape[0]
        collated_samples['diffusion_advantages'] = collated_samples['diffusion_advantages'] / (global_diffusion_reward_std + 1e-6)
        if advantages.ndim == 1:
            advantages = advantages[:, None]

        if advantages.shape[0] == world_size * samples_per_gpu:
            collated_samples["llm_advantages"] = torch.from_numpy(
                advantages.reshape(world_size, samples_per_gpu, -1)[rank]
            ).to(device)
        else:
            assert False

        if accelerator.is_main_process:
            logger.info(f"llm_advantages mean: {collated_samples['llm_advantages'].abs().mean().item()}")
            logger.info(f"diffusion_advantages mean: {collated_samples['diffusion_advantages'].abs().mean().item()}")

        del collated_samples["rewards"]
        del collated_samples["prompt_ids"]

        num_batches = config.sample.num_batches_per_epoch * config.sample.train_batch_size // config.train.batch_size

        filtered_samples = collated_samples

        total_batch_size_filtered, num_timesteps_filtered = filtered_samples["timesteps"].shape

        # ============================================================ #
        #  训练阶段
        # ============================================================ #
        number_mirco_batch = config.inner_diffusion_groupsize // config.mirco_diffusion_train_batch_size

        llm_effective_grad_accum_steps = config.train.gradient_accumulation_steps
        diffusion_effective_grad_accum_steps = config.train.gradient_accumulation_steps * 3 * number_mirco_batch

        diffusion_current_accumulated_steps = 0
        llm_current_accumulated_steps = 0
        gradient_update_times = 0

        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(total_batch_size_filtered, device=device)
            shuffled_filtered_samples = filtered_samples

            train_llm = False
            training_batch_size = total_batch_size_filtered // num_batches

            samples_batched_list = []
            for k_batch in range(num_batches):
                batch_dict = {}
                start = k_batch * training_batch_size
                end = (k_batch + 1) * training_batch_size
                for key, val_tensor in shuffled_filtered_samples.items():
                    if key in ["kv_cache1", "crossattn_cache"]:
                        batch_dict[key] = val_tensor[k_batch]
                    else:
                        batch_dict[key] = val_tensor[start:end]
                samples_batched_list.append(batch_dict)

            info_accumulated = defaultdict(list)

            for i, train_sample_batch in tqdm(
                list(enumerate(samples_batched_list)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_main_process,
            ):
                # -------------------------------------------------- #
                #  LLM 策略训练分支（PPO-style GRPO）
                # -------------------------------------------------- #
                if config.use_llm_lora:
                    llm_loss_terms = {}

                    llm_generation_ids = train_sample_batch["llm_generation_ids"].long()
                    llm_prompt_ids = train_sample_batch["llm_prompt_ids"].long()
                    old_gen_log_probs = train_sample_batch["old_gen_log_probs"]
                    llm_all_advantages = train_sample_batch["llm_advantages"]
                    respones_json_mask = train_sample_batch["respones_json_mask"]
                    llm_advantages_clip = torch.clamp(
                        llm_all_advantages,
                        -config.train.adv_clip_max,
                        config.train.adv_clip_max,
                    )

                    input_ids = torch.cat([llm_prompt_ids, llm_generation_ids], dim=1)
                    attention_mask = (input_ids != llm_policy_tokenizer.pad_token_id).float()
                    # LLM 单独使用 fp16，覆盖外层 bf16 autocast
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = llm_policy(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits.float()

                    logits = logits[:, llm_prompt_ids.shape[1] - 1:-1]

                    labels = llm_generation_ids
                    logits = logits / 1.0
                    log_probs = F.log_softmax(logits, dim=-1)

                    gen_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

                    with torch.no_grad():
                        with unwrapped_llm.disable_adapter():
                            # LLM ref model 也在 fp16 下运行
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                ref_outputs = llm_policy(input_ids=input_ids, attention_mask=attention_mask)
                            ref_logits = ref_outputs.logits.float()
                            ref_logits = ref_logits[:, llm_prompt_ids.shape[1] - 1:-1]
                            ref_logits = ref_logits / 1.0
                            ref_logits = F.log_softmax(ref_logits, dim=-1)
                            ref_gen_log_probs = ref_logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

                    all_response_mask = (labels != llm_policy_tokenizer.pad_token_id).float()
                    response_mask = respones_json_mask

                    if config.gspo:
                        log_ratio = ((gen_log_probs - old_gen_log_probs) * response_mask).sum(-1) / response_mask.sum(-1)
                        ratio = torch.exp(log_ratio)

                        unclip_loss = ratio * llm_advantages_clip
                        clip_loss = torch.clamp(ratio, 1-config.clip_eps, 1+config.clip_eps) * llm_advantages_clip

                        loss = -torch.min(unclip_loss, clip_loss)

                        clip_frac = (torch.abs(ratio.detach() - 1) > config.clip_eps).float()
                        clip_frac_gt = ((ratio.detach() - 1.0) > config.clip_eps).float()
                        clip_frac_lt = ((1.0 - ratio.detach()) > config.clip_eps).float()
                        ratio_mean = ratio.detach()
                        ratio_std = torch.sqrt(
                            (ratio.detach() - 1) ** 2
                        )
                        llm_loss_terms["llm_ratio_mean"] = ratio_mean.mean().detach()
                        llm_loss_terms["llm_ratio_std"] = ratio_std.mean().detach()
                        llm_loss_terms["llm_ratio_max"] = ratio.max().detach()
                        llm_loss_terms["llm_clipfrac"] = clip_frac.mean().detach()
                        llm_loss_terms["llm_clipfrac_gt_one"] = clip_frac_gt.mean().detach()
                        llm_loss_terms["llm_clipfrac_lt_one"] = clip_frac_lt.mean().detach()

                    else:
                        ratio = torch.exp(gen_log_probs - old_gen_log_probs)

                        unclip_loss = ratio * llm_advantages_clip.unsqueeze(-1)
                        clip_loss = torch.clamp(ratio, 1-config.clip_eps, 1+config.clip_eps) * llm_advantages_clip.unsqueeze(-1)

                        loss = -torch.min(unclip_loss, clip_loss)

                        loss = loss * response_mask
                        loss = loss.sum(-1) / response_mask.sum(-1)

                        clip_frac = ((torch.abs(ratio.detach() - 1) > config.clip_eps) * response_mask).sum(-1) / response_mask.sum(-1)
                        clip_frac_gt = (((ratio.detach() - 1.0) > config.clip_eps) * response_mask).sum(-1) / response_mask.sum(-1)
                        clip_frac_lt = (((1.0 - ratio.detach()) > config.clip_eps) * response_mask).sum(-1) / response_mask.sum(-1)
                        ratio_mean = ((ratio.detach() * response_mask).sum(-1) / response_mask.sum(-1)).detach()
                        ratio_std = torch.sqrt(
                            (((ratio.detach() - 1) ** 2) * response_mask).sum(-1) / response_mask.sum(-1)
                        )
                        llm_loss_terms["llm_ratio_mean"] = ratio_mean.mean().detach()
                        llm_loss_terms["llm_ratio_std"] = ratio_std.mean().detach()
                        llm_loss_terms["llm_ratio_max"] = (ratio * response_mask).max().detach()
                        llm_loss_terms["llm_clipfrac"] = clip_frac.mean().detach()
                        llm_loss_terms["llm_clipfrac_gt_one"] = clip_frac_gt.mean().detach()
                        llm_loss_terms["llm_clipfrac_lt_one"] = clip_frac_lt.mean().detach()

                    llm_loss_terms["llm_reponse_length"] = all_response_mask.sum(-1).mean().detach()
                    llm_loss_terms["llm_response_json_length"] = response_mask.sum(-1).mean().detach()

                    llm_loss_terms["llm_policy_loss"] = loss.mean().detach()
                    per_token_kl = torch.exp(ref_gen_log_probs - gen_log_probs) - (ref_gen_log_probs - gen_log_probs) - 1
                    per_token_kl = (per_token_kl * response_mask).sum(-1) / response_mask.sum(-1)

                    llm_loss_terms["llm_kl_loss"] = per_token_kl.mean().detach()

                    loss = loss + per_token_kl * config.train.llm_kl_beta
                    llm_loss_terms["llm_total_loss"] = loss.mean().detach()

                    scaled_loss = loss.mean() / llm_effective_grad_accum_steps
                    # accelerator.backward 替代 scaler.scale(loss).backward() / loss.backward()
                    accelerator.backward(scaled_loss)
                    llm_current_accumulated_steps += 1

                    for k_info, v_info in llm_loss_terms.items():
                        info_accumulated[k_info].append(v_info)

                # -------------------------------------------------- #
                #  Diffusion Transformer 训练分支（NFT 前向过程优化）
                # -------------------------------------------------- #
                num_blocks = len(range(0, diffusion_sample_frame + 3, 3))
                kv_cache1 = pipeline._initialize_kv_cache(
                    batch_size=training_batch_size,
                    num_blocks=num_blocks,
                    dtype=train_sample_batch["latents_clean"].dtype,
                    device=train_sample_batch["latents_clean"].device,
                    return_c=True
                )

                for frame_chunk_idx, frame_chunk_orig_idx in tqdm(
                    enumerate(range(0, diffusion_sample_frame + 3, 3)),
                    desc=f"frame chunk indx {diffusion_sample_frame // 3 + 1}: cache context feature",
                    position=1,
                    disable=not accelerator.is_main_process,
                ):
                    
                    current_start_frame = frame_chunk_orig_idx
                    if current_start_frame > 9:
                        cache_decision = torch.ones([training_batch_size, current_start_frame], device=train_sample_batch["latents_clean"].device, dtype=train_sample_batch["latents_clean"].dtype)
                        cache_decision[:, 1:-8] = 0
                    else:
                        cache_decision = None

                    if llm_policy is not None and current_start_frame % config.gap_frame == 0:
                        prompt_step_idx = current_start_frame // config.gap_frame
                        prompt_embeds_for = {
                            "prompt_embeds": train_sample_batch["prompt_embeds"][:, :, :, prompt_step_idx]
                        }
                        crossattn_cache = pipeline._initialize_crossattn_cache(
                            batch_size=training_batch_size,
                            dtype=train_sample_batch["latents_clean"].dtype,
                            device=train_sample_batch["latents_clean"].device,
                            return_c=True
                        )
                    if current_start_frame != diffusion_sample_frame:
                        # cache context feature in kv_cache1
                        x0 = train_sample_batch["latents_clean"][:, frame_chunk_orig_idx:frame_chunk_orig_idx + 3, :, :, :]
                        input_timestep = pipeline.denoising_step_list[0].to(device=device).view(-1, 1).repeat(1, 3)
                        with torch.no_grad():
                            with accelerator.autocast():
                                context_timestep = torch.ones_like(input_timestep) * pipeline.args.context_noise
                                transformer(
                                    x0.permute(0, 2, 1, 3, 4),
                                    t=context_timestep, context=prompt_embeds_for["prompt_embeds"],
                                    seq_len=pipeline.generator.seq_len,
                                    kv_cache=kv_cache1,
                                    crossattn_cache=crossattn_cache,
                                    current_start=current_start_frame * pipeline.frame_seq_length,
                                    cache_decision=cache_decision
                                ).permute(0, 2, 1, 3, 4)
                    else:
                        # training diffusion
                        highest_x0 = train_sample_batch["highest_latents"].detach() # B, T, F, C, H, W
                        lowest_x0 = train_sample_batch["lowest_latents"].detach() # B, T, F, C, H, W
                        advantages_clip = torch.clamp(
                            train_sample_batch["diffusion_advantages"], # B, G
                            -config.train.diffusion_adv_clip_max,
                            config.train.diffusion_adv_clip_max,
                        )
                        highest_indice = train_sample_batch["highest_indice"]
                        lowest_indice = train_sample_batch["lowest_indice"]
                        advantages_highest = advantages_clip[torch.arange(training_batch_size), highest_indice].unsqueeze(-1).repeat(1, config.mirco_diffusion_train_batch_size).reshape(-1) # B
                        advantages_lowest = advantages_clip[torch.arange(training_batch_size), lowest_indice].unsqueeze(-1).repeat(1, config.mirco_diffusion_train_batch_size).reshape(-1) # B

                        for mirco_index in range(number_mirco_batch):
                            # B * M, T, F, C, H, W
                            train_latents = train_sample_batch["diffusion_train_latents"][:, mirco_index * config.mirco_diffusion_train_batch_size:(mirco_index + 1) * config.mirco_diffusion_train_batch_size].reshape(-1, *train_sample_batch["diffusion_train_latents"].shape[2:])
                            diffusion_old_flow_preds = train_sample_batch["diffusion_flow_preds"][:, mirco_index * config.mirco_diffusion_train_batch_size:(mirco_index + 1) * config.mirco_diffusion_train_batch_size].reshape(-1, *train_sample_batch["diffusion_flow_preds"].shape[2:])
                            # B * M, T-1
                            diffusion_old_log_probs = train_sample_batch["diffusion_log_prob"][:, mirco_index * config.mirco_diffusion_train_batch_size:(mirco_index + 1) * config.mirco_diffusion_train_batch_size].reshape(-1, *train_sample_batch["diffusion_log_prob"].shape[2:])
                            # B, M
                            diffusion_advantages = advantages_clip[:, mirco_index * config.mirco_diffusion_train_batch_size:(mirco_index + 1) * config.mirco_diffusion_train_batch_size].reshape(-1)

                            for j_idx, j_timestep_orig in tqdm(
                                enumerate(pipeline.denoising_step_list[:-1]),
                                desc="Timestep",
                                position=2,
                                leave=False,
                                disable=not accelerator.is_main_process,
                            ):
                                j_timestep_orig = pipeline.denoising_step_list[j_idx]
                                xt = train_latents[:, j_idx, :, :, :, :]
                                xt_1 = train_latents[:, j_idx + 1, :, :, :, :]
                                diffusion_old_flow_pred =  diffusion_old_flow_preds[:, j_idx, :, :, :, :]
                                diffusion_old_log_prob =  diffusion_old_log_probs[:, j_idx, :, :, :, :]
                                highest_x0_step = highest_x0[:, j_idx + 1, :, :, :, :]
                                lowest_x0_step = lowest_x0[:, j_idx + 1, :, :, :, :]

                                input_timestep = j_timestep_orig.to(device=device).view(-1, 1).repeat(1, 3)
                                sigma_t = j_timestep_orig.float() / 1000
                                sigma_t_1 = pipeline.denoising_step_list[j_idx + 1].float() / 1000

                                with accelerator.autocast():
                                    # -- default adapter（当前策略，需要梯度）--
                                    forward_prediction = transformer(
                                        xt.permute(0, 2, 1, 3, 4),
                                        t=input_timestep, context=prompt_embeds_for["prompt_embeds"],
                                        seq_len=pipeline.generator.seq_len,
                                        kv_cache=kv_cache1,
                                        crossattn_cache=crossattn_cache,
                                        current_start=current_start_frame * pipeline.frame_seq_length,
                                        cache_decision=cache_decision,
                                        recording_cache=False
                                    ).permute(0, 2, 1, 3, 4)

                                    # -- 参考模型（disable adapter，无梯度）--
                                    with torch.no_grad():
                                        if config.use_lora:
                                            with unwrapped_transformer.disable_adapter():
                                                ref_forward_prediction = transformer(
                                                    xt.permute(0, 2, 1, 3, 4),
                                                    t=input_timestep, context=prompt_embeds_for["prompt_embeds"],
                                                    seq_len=pipeline.generator.seq_len,
                                                    kv_cache=kv_cache1,
                                                    crossattn_cache=crossattn_cache,
                                                    current_start=current_start_frame * pipeline.frame_seq_length,
                                                    cache_decision=cache_decision,
                                                    recording_cache=False
                                                ).permute(0, 2, 1, 3, 4)
                                            unwrapped_transformer.set_adapter("default")
                                        else:
                                            assert False

                                diffusion_loss_terms = {}

                                # ----------main policy loss-------------
                                current_log_prob = compute_log_prob(
                                    xt.float(), # x_t
                                    xt_1.float(), # x_t_1
                                    forward_prediction.float(), 
                                    sigma_t, 
                                    sigma_t_1
                                )
                                if config.ratio_norm:
                                    ratio_bias_scale =  ((1 - sigma_t_1) ** 2) * (sigma_t ** 2) / (2 * sigma_t_1 ** 2)
                                    v_prediction_detla = (forward_prediction.float() - diffusion_old_flow_pred.float()).pow(2).mean(dim=tuple(range(1, forward_prediction.ndim)))
                                    log_ratio = current_log_prob.mean(dim=tuple(range(1, current_log_prob.ndim))) - diffusion_old_log_prob.mean(dim=tuple(range(1, diffusion_old_log_prob.ndim)))
                                    ratio = torch.exp((log_ratio + ratio_bias_scale * v_prediction_detla) * sigma_t_1 / ((1 - sigma_t_1) * sigma_t))
                                    diffusion_loss_terms["log_ratio"] = log_ratio.mean().detach()
                                    diffusion_loss_terms["v_prediction_detla"] = v_prediction_detla.mean().detach()
                                    diffusion_loss_terms[f"step{j_idx}_ratio_shift"] = (log_ratio + ratio_bias_scale * v_prediction_detla).mean().detach()
                                    diffusion_loss_terms[f"step{j_idx}_ratio"] = ratio.mean().detach() - 1
                                    diffusion_loss_terms[f"step{j_idx}_ratio_bias_scale"] = ratio_bias_scale.detach()
                                    diffusion_loss_terms[f"step{j_idx}_ratio_norm_scale"] = sigma_t_1 / ((1 - sigma_t_1) * sigma_t).detach()

                                else:
                                    ratio = torch.exp(current_log_prob.mean(dim=tuple(range(1, current_log_prob.ndim))) - diffusion_old_log_prob.mean(dim=tuple(range(1, diffusion_old_log_prob.ndim))))
                                unclipped_loss = -diffusion_advantages * ratio
                                clipped_loss = -diffusion_advantages * torch.clamp(
                                    ratio,
                                    1.0 - config.train.clip_range,
                                    1.0 + config.train.clip_range,
                                )
                                policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                                diffusion_loss_terms["diffusion_policy_loss"] = policy_loss.detach()
                                if config.reweight and not config.ratio_norm:
                                    policy_loss = sigma_t_1 * policy_loss / ((1 - sigma_t_1) * sigma_t)
                                    diffusion_loss_terms["diffusion_reweight_policy_loss"] = policy_loss.detach()

                                diffusion_loss_terms[f"step{j_idx}_diffusion_clipfrac"] = torch.mean((torch.abs(ratio - 1.0) > config.train.clip_range).float()).detach()
                                diffusion_loss_terms[f"step{j_idx}_diffusion_clipfrac_gt_one"] = torch.mean((ratio - 1.0 > config.train.clip_range).float()).detach()
                                diffusion_loss_terms[f"step{j_idx}_diffusion_clipfrac_lt_one"] = torch.mean((ratio - 1.0 < -config.train.clip_range).float()).detach()

                                # -----------aux policy loss--------------
                                if config.aux_loss_beta > 0:
                                    highest_log_prob = compute_log_prob(
                                        xt.float(), # x_t
                                        highest_x0_step.float(), # x_t_1
                                        forward_prediction.float(), 
                                        sigma_t, 
                                        sigma_t_1
                                    )

                                    highest_old_log_prob = compute_log_prob(
                                        xt.float(), # x_t
                                        highest_x0_step.float(), # x_t_1
                                        diffusion_old_flow_pred.float(), 
                                        sigma_t, 
                                        sigma_t_1
                                    )

                                    highest_ratio = torch.exp(highest_log_prob - highest_old_log_prob)
                                    hight_unclipped_loss = -advantages_highest * highest_ratio
                                    high_clipped_loss = -advantages_highest * torch.clamp(
                                        highest_ratio,
                                        1.0 - config.train.clip_range,
                                        1.0 + config.train.clip_range,
                                    )
                                    high_policy_loss = torch.mean(torch.maximum(hight_unclipped_loss, high_clipped_loss))
                                    diffusion_loss_terms["diffusion_high_policy_loss"] = high_policy_loss.detach()
                                    diffusion_loss_terms["diffusion_high_clipfrac"] = torch.mean((torch.abs(highest_ratio - 1.0) > config.train.clip_range).float()).detach()
                                    diffusion_loss_terms["diffusion_high_clipfrac_gt_one"] = torch.mean((highest_ratio - 1.0 > config.train.clip_range).float()).detach()
                                    diffusion_loss_terms["diffusion_high_clipfrac_lt_one"] = torch.mean((highest_ratio - 1.0 < -config.train.clip_range).float()).detach()

                                    lowest_log_prob = compute_log_prob(
                                        xt.float(), # x_t
                                        lowest_x0_step.float(), # x_t_1
                                        forward_prediction.float(), 
                                        sigma_t, 
                                        sigma_t_1
                                    )

                                    lowest_old_log_prob = compute_log_prob(
                                        xt.float(), # x_t
                                        lowest_x0_step.float(), # x_t_1
                                        diffusion_old_flow_pred.float(), 
                                        sigma_t, 
                                        sigma_t_1
                                    )
                                    lowest_ratio = torch.exp(lowest_log_prob - lowest_old_log_prob)
                                    lowest_unclipped_loss = -advantages_lowest * lowest_ratio
                                    lowest_clipped_loss = -advantages_lowest * torch.clamp(
                                        lowest_ratio,
                                        1.0 - config.train.clip_range,
                                        1.0 + config.train.clip_range,
                                    )
                                    low_policy_loss = torch.mean(torch.maximum(lowest_unclipped_loss, lowest_clipped_loss))
                                    diffusion_loss_terms["diffusion_low_policy_loss"] = low_policy_loss.detach()
                                    diffusion_loss_terms["diffusion_low_clipfrac"] = torch.mean((torch.abs(lowest_ratio - 1.0) > config.train.clip_range).float()).detach()
                                    diffusion_loss_terms["diffusion_low_clipfrac_gt_one"] = torch.mean((lowest_ratio - 1.0 > config.train.clip_range).float()).detach()
                                    diffusion_loss_terms["diffusion_low_clipfrac_lt_one"] = torch.mean((lowest_ratio - 1.0 < -config.train.clip_range).float()).detach()

                                    loss = policy_loss + config.aux_loss_beta * (high_policy_loss + low_policy_loss)
                                else:
                                    loss = policy_loss


                                # -- KL 散度损失 --
                                kl_div_loss = ((forward_prediction - ref_forward_prediction) ** 2).mean(
                                    dim=tuple(range(1, x0.ndim))
                                )
                                if (diffusion_current_accumulated_steps // diffusion_effective_grad_accum_steps) % 2 == 0:
                                    loss += config.train.diffusion_kl_beta * torch.mean(kl_div_loss)
                                    kl_div_loss = torch.mean(kl_div_loss)
                                    diffusion_loss_terms["diffusion_kl_div_loss"] = torch.mean(kl_div_loss).detach()
                                    diffusion_loss_terms["diffusion_kl_div"] = torch.mean(
                                        ((forward_prediction - ref_forward_prediction) ** 2).mean(dim=tuple(range(1, x0.ndim)))
                                    ).detach()
                                    
                                diffusion_loss_terms["diffusion_total_loss"] = loss.detach()

                                # -- 反向传播（accelerator.backward 替代 scaler/backward）--
                                scaled_loss = loss / diffusion_effective_grad_accum_steps
                                accelerator.backward(scaled_loss)
                                diffusion_current_accumulated_steps += 1

                                for k_info, v_info in diffusion_loss_terms.items():
                                    info_accumulated[k_info].append(v_info)

                                if (diffusion_current_accumulated_steps % diffusion_effective_grad_accum_steps == 0 and \
                                    llm_current_accumulated_steps % llm_effective_grad_accum_steps == 0):
                                    # accelerator.clip_grad_norm_ 内部处理 unscale + clip
                                    params_to_clip = [
                                        p
                                        for group in optimizer.param_groups
                                        for p in group["params"]
                                        if p.requires_grad and p.grad is not None
                                    ]
                                    accelerator.clip_grad_norm_(params_to_clip, config.train.max_grad_norm)

                                    optimizer.step()
                                    gradient_update_times += 1
                                    optimizer.zero_grad()

                                    log_info = {k: torch.mean(torch.stack(v_list)).item() for k, v_list in info_accumulated.items()}
                                    info_tensor = torch.tensor([log_info[k] for k in sorted(log_info.keys())], device=device)
                                    # accelerator.reduce 替代 dist.all_reduce(..., AVG)
                                    info_tensor = accelerator.reduce(info_tensor, reduction="mean")
                                    reduced_log_info = {k: info_tensor[ki].item() for ki, k in enumerate(sorted(log_info.keys()))}
                                    if accelerator.is_main_process:
                                        wandb.log(
                                            {
                                                "step": global_step,
                                                "gradient_update_times": gradient_update_times,
                                                "epoch": epoch,
                                                "inner_epoch": inner_epoch,
                                                **reduced_log_info,
                                            }
                                        )

                                    global_step += 1
                                    info_accumulated = defaultdict(list)

                        if (
                            config.train.ema
                            and ema is not None
                            and (diffusion_current_accumulated_steps % diffusion_effective_grad_accum_steps == 0)
                        ):
                            ema.step(transformer_trainable_parameters, global_step)

                del crossattn_cache
                del kv_cache1

        # ---- epoch 结束：同步 barrier ----
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        wandb.finish()
    # accelerator.end_training() 用于释放资源（可选，accelerate 会自动清理）
    accelerator.end_training()


if __name__ == "__main__":
    app.run(main)
