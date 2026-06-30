"""
Unified inference script for Self-Forcing and LongLive video generation models.

Usage
-----
Single GPU:
    python tools/inference_unified.py \\
        --mode self_forcing \\
        --config_path self_forcing/config/self_forcing_dmd.yaml \\
        --model_path /path/to/self_forcing_dmd.pt \\
        --lora_path  /path/to/checkpoint-320 \\
        --output_file /path/to/output/dir

Multi-GPU (e.g. 4 cards):
    torchrun --nproc_per_node=4 tools/inference_unified.py \\
        --mode longlive \\
        --config_path self_forcing/config/longlive_interactive_inference.yaml \\
        --lora_path  /path/to/checkpoint-320 \\
        --output_file /path/to/output/dir

Model modes (--mode)
--------------------
self_forcing
    Pipeline : CausalInferencePipeline
    Sampling : pipeline.inference_withllmpolicy_GRPO_sample
               (unified for both llm-only and llm+diffusion-mix variants;
               distinction is purely whether <lora_path>/lora/ folder is present)

longlive
    Pipeline : InteractiveCausalInferencePipeline
    Extra    : loads & merges longlive LoRA from config before anything else
    Sampling : pipeline.grpo_llm_inference (regardless of diffusion LoRA)

LoRA auto-detection (<lora_path> is the checkpoint directory)
-------------------------------------------------------------
Diffusion LoRA : loaded when  <lora_path>/lora/      directory exists
LLM LoRA       : loaded from  <lora_path>/llm_lora/  when present;
                 otherwise falls back to --llm_fallback_lora_path
                 (the cold-start LLM planner baseline, ckpt/pre_llm_policy_lora)

Multi-GPU
---------
Prompts are round-robin partitioned across WORLD_SIZE ranks.
Each rank writes inference_results_rank{N}.csv.
Rank 0 merges all per-rank CSVs into inference_results.csv after the barrier.
"""

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
import peft
import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import OmegaConf
from peft import PeftModel
from tqdm import tqdm
from torchvision.io import write_video
from transformers import AutoModelForCausalLM, AutoTokenizer

from self_forcing.causal_pipeline import CausalInferencePipeline
from self_forcing.causal_pipeline_longlive import CausalInferencePipeline as CausalInferencePipelineLonglive
from self_forcing.interactive_causal_pipeline import InteractiveCausalInferencePipeline
from self_forcing.lora_utils import configure_lora_for_model
from self_forcing.policy_model_v2 import MemoryPolicyROPE

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = True
torch._inductor.config.debug = True

# ─────────────────────────────────────────────────────────────────────────────
# Default paths
# ─────────────────────────────────────────────────────────────────────────────

_LLM_BASE_MODEL = (
    "/path/to/hf_cache/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
)

_LLM_FALLBACK_LORA = (
    "ckpt/pre_llm_policy_lora"
)

_DEFAULT_CONFIG = (
    "self_forcing/config/default_config.yaml"
)

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed():
    """
    Read RANK / LOCAL_RANK / WORLD_SIZE from env (set by torchrun),
    initialise the NCCL process group when world_size > 1, and pin
    the current process to its GPU.

    Returns (local_rank, world_size, rank).
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank       = int(os.environ.get("RANK",       0))

    if world_size > 1:
        dist.init_process_group(backend="nccl")

    torch.cuda.set_device(local_rank)
    return local_rank, world_size, rank


def partition_prompts(df: pd.DataFrame, rank: int, world_size: int) -> pd.DataFrame:
    """Round-robin slice of *df* belonging to *rank*, preserving original index."""
    return df.iloc[list(range(rank, len(df), world_size))].reset_index(drop=False)


def _append_csv(path: str, rows: list, write_header: bool) -> None:
    pd.DataFrame(rows).to_csv(path, mode="a", header=write_header, index=False)


def merge_rank_csvs(output_file: str, world_size: int) -> str:
    """
    Collect inference_results_rank{N}.csv written by each rank, sort by the
    original prompt index, write a single inference_results.csv, and remove
    the per-rank files.  Called only on rank 0 after the distributed barrier.
    """
    frames = []
    for r in range(world_size):
        rank_csv = os.path.join(output_file, f"inference_results_rank{r}.csv")
        if os.path.exists(rank_csv):
            frames.append(pd.read_csv(rank_csv))

    if not frames:
        print("[merge] No per-rank CSVs found; nothing to merge.")
        return ""

    merged = pd.concat(frames, ignore_index=True)
    if "index" in merged.columns:
        merged = merged.sort_values("index").reset_index(drop=True)

    merged_path = os.path.join(output_file, "inference_results.csv")
    merged.to_csv(merged_path, index=False)
    print(f"[merge] Wrote {len(merged)} rows → {merged_path}")

    # Clean up per-rank files
    for r in range(world_size):
        rank_csv = os.path.join(output_file, f"inference_results_rank{r}.csv")
        if os.path.exists(rank_csv):
            os.remove(rank_csv)

    return merged_path


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline builders
# ─────────────────────────────────────────────────────────────────────────────

def build_self_forcing_pipeline(
    config,
    model_path: str,
    device: str,
) -> CausalInferencePipeline:
    """
    Instantiate CausalInferencePipeline and load the base generator weights
    from *model_path* (a .pt file with a 'generator_ema' key).
    """
    pipeline = CausalInferencePipeline(config, device=device)

    if model_path and os.path.isfile(model_path):
        state_dict = torch.load(model_path, map_location="cpu")
        pipeline.generator.load_state_dict(state_dict["generator_ema"])
        print(f"[self_forcing] Loaded base weights from {model_path}")
    else:
        print("[self_forcing] No model_path given; using randomly initialised weights.")

    return pipeline


def build_longlive_pipeline(
    config,
    device: str,
) -> InteractiveCausalInferencePipeline:
    """
    Instantiate InteractiveCausalInferencePipeline, load the longlive
    checkpoint (EMA or plain), apply the LoRA adapter defined in config,
    load the config-specified LoRA weights, then **merge and unload** the
    LoRA into the base model so subsequent calls see a clean dense model.
    """
    pipeline = InteractiveCausalInferencePipeline(config, device=device)

    # ── 1. Load base / EMA weights ────────────────────────────────────────
    generator_ckpt = getattr(config, "generator_ckpt", None)
    if generator_ckpt and os.path.isfile(generator_ckpt):
        state_dict = torch.load(generator_ckpt, map_location="cpu")
        use_ema    = getattr(config, "use_ema", True)
        raw        = state_dict["generator_ema" if use_ema else "generator"]

        if use_ema:
            def _strip_fsdp(name: str) -> str:
                return name.replace("_fsdp_wrapped_module.", "")
            raw = {_strip_fsdp(k): v for k, v in raw.items()}
            pipeline.generator.load_state_dict(raw, strict=False)
        else:
            pipeline.generator.load_state_dict(raw)

        print(f"[longlive] Loaded checkpoint from {generator_ckpt}")
    else:
        print("[longlive] No generator_ckpt in config; using randomly initialised weights.")

    # ── 2. Apply config LoRA structure, load weights, then merge ──────────
    pipeline.is_lora_enabled = False
    if getattr(config, "adapter", None):
        pipeline.generator.model = configure_lora_for_model(
            pipeline.generator.model,
            model_name="generator",
            lora_config=config.adapter,
            is_main_process=True,
        )

        lora_ckpt_path = getattr(config, "lora_ckpt", None)
        if lora_ckpt_path and os.path.isfile(lora_ckpt_path):
            print(f"[longlive] Loading LoRA weights from {lora_ckpt_path}")
            ckpt = torch.load(lora_ckpt_path, map_location="cpu")
            if isinstance(ckpt, dict) and "generator_lora" in ckpt:
                peft.set_peft_model_state_dict(
                    pipeline.generator.model, ckpt["generator_lora"]
                )
            else:
                peft.set_peft_model_state_dict(pipeline.generator.model, ckpt)

            print("[longlive] Merging longlive LoRA into base model ...")
            pipeline.generator.model = pipeline.generator.model.merge_and_unload()
            print("[longlive] Merge complete.")

        pipeline.is_lora_enabled = True

    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# LoRA loading helpers  (shared by both modes)
# ─────────────────────────────────────────────────────────────────────────────

def maybe_load_diffusion_lora(pipeline, lora_path: Optional[str]) -> None:
    """
    If <lora_path>/lora/ exists, wrap the generator transformer with it as a
    PeftModel adapter.
    """
    if not lora_path:
        print("[diffusion_lora] No lora_path provided; using base diffusion weights.")
        return

    diffusion_lora_dir = os.path.join(lora_path, "lora")
    if os.path.isdir(diffusion_lora_dir):
        print(f"[diffusion_lora] Loading from {diffusion_lora_dir}")
        pipeline.generator.model = PeftModel.from_pretrained(
            pipeline.generator.model, diffusion_lora_dir
        )
        pipeline.generator.model.set_adapter("default")
    else:
        print(f"[diffusion_lora] {diffusion_lora_dir} not found; "
              "using base diffusion weights.")


def load_llm_with_lora(lora_path: Optional[str], fallback_lora_path: str):
    """
    Load the Qwen3-1.7B LLM base and attach a LoRA adapter:
      - <lora_path>/llm_lora/          if it exists,
      - <fallback_lora_path>/llm_lora/ otherwise.

    Returns (llm_model, tokenizer).
    """
    print(f"[llm] Loading base LLM from {_LLM_BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(_LLM_BASE_MODEL)
    llm = AutoModelForCausalLM.from_pretrained(
        _LLM_BASE_MODEL,
        torch_dtype="auto",
        device_map="auto",
    )

    chosen_lora_dir = None
    if lora_path:
        candidate = os.path.join(lora_path, "llm_lora")
        if os.path.isdir(candidate):
            chosen_lora_dir = candidate
            print(f"[llm] Using trained LLM LoRA from {chosen_lora_dir}")

    if chosen_lora_dir is None:
        # Fallback (cold-start baseline). The adapter may live either directly
        # in fallback_lora_path or under a llm_lora/ sub-folder.
        candidate = os.path.join(fallback_lora_path, "llm_lora")
        chosen_lora_dir = candidate if os.path.isdir(candidate) else fallback_lora_path
        print(f"[llm] llm_lora not found in lora_path; "
              f"falling back to {chosen_lora_dir}")

    llm = PeftModel.from_pretrained(llm, chosen_lora_dir)
    llm.set_adapter("default")
    return llm, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Inference runners
# ─────────────────────────────────────────────────────────────────────────────

def run_self_forcing(
    pipeline: CausalInferencePipeline,
    llm,
    tokenizer,
    prompts_df: pd.DataFrame,
    output_file: str,
    dtype: torch.dtype,
    sample_frames: int,
    gap_frame: int,
    policy_path: Optional[str],
    seed: int,
    rank: int,
) -> str:
    """
    Inference loop for self_forcing mode.

    Sampling always uses pipeline.inference_withllmpolicy_GRPO_sample.
    Whether diffusion LoRA is active is already reflected in the pipeline
    by the time this function is called (via maybe_load_diffusion_lora).
    One video is generated per prompt.
    """
    os.makedirs(output_file, exist_ok=True)

    # ── Policy model (lightweight, ROPE-based) ─────────────────────────────
    policy_model = MemoryPolicyROPE(
        dim=pipeline.generator.model.dim,
        anchor_front=1,
        anchor_back=1,
    ).to(device="cuda", dtype=dtype)

    if policy_path and os.path.isfile(policy_path):
        policy_model.load_state_dict(torch.load(policy_path, map_location="cuda"))
        print(f"[policy] Loaded policy weights from {policy_path}")
    else:
        print("[policy] No policy_path; using randomly initialised policy.")

    # ── Move pipeline to device ────────────────────────────────────────────
    pipeline = pipeline.to(dtype=dtype)
    pipeline.generator.to(device="cuda", dtype=dtype)
    pipeline.vae.to(device="cuda")
    pipeline.text_encoder.to(device="cuda", dtype=dtype)
    pipeline.text_encoder.eval()

    csv_path   = os.path.join(output_file, f"inference_results_rank{rank}.csv")
    has_header = not os.path.exists(csv_path)
    results    = []

    for _, row in prompts_df.iterrows():
        df_idx  = int(row["index"]) if "index" in row.index else int(row.name)
        out_path = os.path.join(output_file, f"index{df_idx}_self_forcing.mp4")

        if os.path.exists(out_path):
            print(f"[rank {rank}] Skipping index {df_idx} (already generated)")
            continue

        caption = row["prompt"]
        print(f"[rank {rank}] index {df_idx}: {caption[:100]}")
        save_dict = {"index": df_idx, "caption": caption}

        set_seed(seed)
        noise = torch.randn(
            [1, sample_frames, 16, 60, 104],
            device="cuda", dtype=dtype,
        )

        only_one_prompt = False
        if only_one_prompt:
            collate_data = pipeline.inference_withllmpolicy_GRPO_sample(
                noise=noise,
                text_prompts=[caption],
                policy_model=policy_model,
                llm_semantic_policy=None,
                llm_policy_tokenizer=tokenizer,
                low_memory=False,
                return_log_prob=False,
                gap_frame=gap_frame,
            )
        else:
            collate_data = pipeline.inference_withllmpolicy_GRPO_sample(
                noise=noise,
                text_prompts=[caption],
                policy_model=policy_model,
                llm_semantic_policy=llm,
                llm_policy_tokenizer=tokenizer,
                low_memory=False,
                return_log_prob=False,
                gap_frame=gap_frame,
            )

        prompt_for_reward = collate_data.get("prompt_for_reward", caption)
        # video: (1, T, C, H, W)  →  (T, H, W, C)
        current_video = rearrange(
            collate_data["video"], "b t c h w -> b t h w c"
        ).cpu()
        pipeline.vae.model.clear_cache()

        if isinstance(prompt_for_reward, (list, tuple)):
            prompt_for_reward = prompt_for_reward[0]
        save_dict["prompt_for_reward"] = prompt_for_reward

        write_video(out_path, 255.0 * current_video[0], fps=16)
        save_dict["video"] = out_path

        results.append(save_dict)
        _append_csv(csv_path, results, write_header=has_header)
        results    = []
        has_header = False

    return csv_path


def run_longlive(
    pipeline: InteractiveCausalInferencePipeline,
    llm,
    tokenizer,
    prompts_df: pd.DataFrame,
    output_file: str,
    dtype: torch.dtype,
    sample_frames: int,
    gap_frame: int,
    seed: int,
    rank: int,
) -> str:
    """
    Inference loop for longlive mode.

    Always uses pipeline.grpo_llm_inference regardless of whether a
    diffusion LoRA is loaded.  One video is generated per prompt.
    """
    os.makedirs(output_file, exist_ok=True)

    pipeline = pipeline.to(dtype=dtype)
    pipeline.generator.to(device="cuda", dtype=dtype)
    pipeline.vae.to(device="cuda")
    pipeline.text_encoder.to(device="cuda", dtype=dtype)
    pipeline.text_encoder.eval()

    csv_path   = os.path.join(output_file, f"inference_results_rank{rank}.csv")
    has_header = not os.path.exists(csv_path)
    results    = []

    for _, row in prompts_df.iterrows():
        df_idx   = int(row["index"]) if "index" in row.index else int(row.name)
        out_path = os.path.join(output_file, f"index{df_idx}_longlive.mp4")

        if os.path.exists(out_path):
            print(f"[rank {rank}] Skipping index {df_idx} (already generated)")
            continue

        caption = row["prompt"]
        print(f"[rank {rank}] index {df_idx}: {caption[:100]}")
        save_dict = {"index": df_idx, "caption": caption}

        set_seed(seed)
        noise = torch.randn(
            [1, sample_frames, 16, 60, 104],
            device="cuda", dtype=dtype,
        )

        collate_data = pipeline.grpo_llm_inference(
            noise=noise,
            text_prompts_list=[caption],
            low_memory=False,
            llm_semantic_policy=llm,
            llm_policy_tokenizer=tokenizer,
            gap_frame=gap_frame,
            return_log_prob=False,
        )

        # video: (1, T, C, H, W)  →  (T, H, W, C)
        current_video = rearrange(
            collate_data["video"], "b t c h w -> b t h w c"
        ).cpu()
        pipeline.vae.model.clear_cache()

        save_dict["prompt_for_reward"] = collate_data.get("prompt_for_reward")

        write_video(out_path, 255.0 * current_video[0], fps=16)
        save_dict["video"] = out_path

        results.append(save_dict)
        _append_csv(csv_path, results, write_header=has_header)
        results    = []
        has_header = False

    return csv_path

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified Self-Forcing / LongLive inference (single or multi-GPU)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ──────────────────────────────────────────────────────────────
    parser.add_argument("--prompt", type=str, default="",
                        help="Single prompt string (used when --prompt_path is absent)")
    parser.add_argument("--prompt_path", type=str, default="",
                        help="CSV file with a 'prompt' column")

    # ── Mode ──────────────────────────────────────────────────────────────
    parser.add_argument("--mode", type=str, default="self_forcing",
                        choices=["self_forcing", "longlive"],
                        help="Model mode")

    # ── Paths ─────────────────────────────────────────────────────────────
    parser.add_argument("--config_path", type=str, required=True,
                        help="OmegaConf YAML config for the chosen pipeline")
    parser.add_argument("--default_config_path", type=str, default=_DEFAULT_CONFIG,
                        help="Base default config merged before --config_path")
    parser.add_argument("--model_path", type=str, default="",
                        help="Base .pt checkpoint (self_forcing only; "
                             "longlive uses generator_ckpt from config)")
    parser.add_argument("--lora_path", type=str, default="",
                        help="Checkpoint dir; /lora and /llm_lora sub-folders "
                             "are auto-detected")
    parser.add_argument("--llm_fallback_lora_path", type=str,
                        default=_LLM_FALLBACK_LORA,
                        help="Fallback LLM LoRA used when lora_path/llm_lora is absent")
    parser.add_argument("--policy_path", type=str, default="",
                        help="Optional MemoryPolicyROPE weights (self_forcing only)")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Directory where generated videos and CSV are saved")

    # ── Sampling hyperparameters ───────────────────────────────────────────
    parser.add_argument("--sample_frames", type=int, default=54,
                        help="Number of latent frames (T dimension of noise)")
    parser.add_argument("--gap_frame", type=int, default=18,
                        help="Frame stride between LLM prompt switches")
    parser.add_argument("--fps", type=int, default=16,
                        help="Output video FPS")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16"])

    args  = parser.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    # ── Distributed setup ─────────────────────────────────────────────────
    local_rank, world_size, rank = setup_distributed()
    device = f"cuda:{local_rank}"

    if rank == 0:
        print(f"Distributed  : world_size={world_size}")
        print(f"Mode         : {args.mode}")
        print(f"lora_path    : {args.lora_path or '(none)'}")
        print(f"output_file  : {args.output_file}")

    # ── Load & partition prompts ───────────────────────────────────────────
    if args.prompt_path and os.path.exists(args.prompt_path):
        if args.prompt_path.endswith(".txt"):
            with open(args.prompt_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            prompts_df = pd.DataFrame([{"prompt": p} for p in lines])
        else:
            prompts_df = pd.read_csv(args.prompt_path)
    elif args.prompt:
        prompts_df = pd.DataFrame([{"prompt": args.prompt}])
    else:
        raise ValueError("Provide at least one of --prompt or --prompt_path")

    my_prompts = partition_prompts(prompts_df, rank, world_size)

    # ── Merge configs ─────────────────────────────────────────────────────
    config = OmegaConf.merge(
        OmegaConf.load(args.default_config_path),
        OmegaConf.load(args.config_path),
    )

    # ── Detect available LoRAs (log on rank 0) ────────────────────────────
    lora_path = args.lora_path or None
    if rank == 0:
        has_diff  = bool(lora_path and os.path.isdir(os.path.join(lora_path, "lora")))
        has_llm   = bool(lora_path and os.path.isdir(os.path.join(lora_path, "llm_lora")))
        print(f"diffusion_lora : {'found' if has_diff else 'not found (using base)'}")
        print(f"llm_lora       : {'found' if has_llm  else 'not found (using fallback)'}")

    # ── Build pipeline ────────────────────────────────────────────────────
    torch.set_grad_enabled(False)

    if args.mode == "self_forcing":
        pipeline = build_self_forcing_pipeline(config, args.model_path, device)
    else:
        pipeline = build_longlive_pipeline(config, device)

    # ── Load diffusion LoRA (shared) ──────────────────────────────────────
    maybe_load_diffusion_lora(pipeline, lora_path)

    # ── Load LLM + LoRA (shared) ──────────────────────────────────────────
    llm, tokenizer = load_llm_with_lora(lora_path, args.llm_fallback_lora_path)

    # ── Run inference ─────────────────────────────────────────────────────
    if args.mode == "self_forcing":
        csv_path = run_self_forcing(
            pipeline=pipeline,
            llm=llm,
            tokenizer=tokenizer,
            prompts_df=my_prompts,
            output_file=args.output_file,
            dtype=dtype,
            sample_frames=args.sample_frames,
            gap_frame=args.gap_frame,
            policy_path=args.policy_path or None,
            seed=args.seed,
            rank=rank,
        )
    else:
        csv_path = run_longlive(
            pipeline=pipeline,
            llm=llm,
            tokenizer=tokenizer,
            prompts_df=my_prompts,
            output_file=args.output_file,
            dtype=dtype,
            sample_frames=args.sample_frames,
            gap_frame=args.gap_frame,
            seed=args.seed,
            rank=rank,
        )

    print(f"[rank {rank}] Per-rank results → {csv_path}")

    # ── Barrier + merge CSVs on rank 0 ───────────────────────────────────
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        merged_path = merge_rank_csvs(args.output_file, world_size)
        if merged_path:
            print(f"[rank 0] Merged CSV → {merged_path}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
