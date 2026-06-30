import importlib
import os
from omegaconf import OmegaConf

# base = importlib.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))
spec = importlib.util.spec_from_file_location("base", os.path.join(os.path.dirname(__file__), "base.py"))
base = importlib.util.module_from_spec(spec)

spec.loader.exec_module(base)

def get_config(name):
    return globals()[name]()


def _get_longlive_text_grpo_diffusion_flowgrpo_mix(base_model="sd3", n_gpus=1, gradient_step_per_epoch=1, dataset="pickscore", reward_fn={}, name=""):
    config = base.get_config()
    assert base_model in ["sd3", "longlive"]
    assert dataset in ["pickscore", "ocr", "geneval", "temporal_order"]

    config.base_model = base_model
    config.dataset = dataset

    longlive_config = OmegaConf.load("/path/to/TempAct/self_forcing/config/longlive_interactive_inference.yaml")
    default_config = OmegaConf.load("/path/to/TempAct/self_forcing/config/default_config.yaml")
    config.pretrained.self_config = OmegaConf.merge(default_config, longlive_config)

    config.self_model_path = "/path/to/hf_cache/models--gdhe17--Self-Forcing/snapshots/2f8b779212da279d212c22a509b66ad6552f350e/checkpoints/self_forcing_dmd.pt"
    config.llm_model_path = "/path/to/hf_cache/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    # Pre-trained LLM-policy LoRA used to warm-start training (set to your own checkpoint).
    config.train.llm_lora_path = "ckpt/pre_llm_policy_lora"
    config.inner_diffusion_groupsize = 8
    config.mirco_diffusion_train_batch_size= 1
    config.sample.eval_num_steps = 40
    config.sample_frames = 30
    config.train_height = 480
    config.train_width = 832
    config.gap_frame = 6
    bsz = 1

    config.sample.num_image_per_prompt = 4
    num_groups = 16

    while True:
        if bsz < 1:
            assert False, "Cannot find a proper batch size."
        if (
            num_groups * config.sample.num_image_per_prompt % (n_gpus * bsz) == 0
            and bsz * n_gpus % config.sample.num_image_per_prompt == 0
        ):
            n_batch_per_epoch = num_groups * config.sample.num_image_per_prompt // (n_gpus * bsz)
            if n_batch_per_epoch % gradient_step_per_epoch == 0:
                config.sample.train_batch_size = bsz
                config.sample.num_batches_per_epoch = n_batch_per_epoch
                config.train.batch_size = bsz
                config.train.gradient_accumulation_steps = (
                    config.sample.num_batches_per_epoch // gradient_step_per_epoch
                )
                break
        bsz -= 1

    # special design, the test set has a total of 1018/2212/2048 for ocr/geneval/pickscore, to make gpu_num*bs*n as close as possible to it, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.
    config.sample.test_batch_size = 2 if dataset == "geneval" else 2
    if n_gpus > 32:
        config.sample.test_batch_size = config.sample.test_batch_size // 2

    config.prompt_fn = "self_forcing"

    config.run_name = name
    config.save_dir = f"outputs/{list(reward_fn.keys())[0]}_llm_diffusion_mix/{base_model}/{name}"
    config.reward_fn = reward_fn

    # Diffusion config
    config.train.diffusion_adv_clip_max = 2.0
    config.train.learning_rate = 1e-4
    config.aux_loss_beta = 0.0
    config.train.clip_range = 1e-5
    config.train.ema = True
    config.train.diffusion_kl_beta = 0.0005
    config.diffusion_sample_frames = [6, 12]
    config.reweight = True
    config.ratio_norm = True
    config.train.ema_decay = 0.99

    # llm config
    config.train.llm_learning_rate = 1e-5
    config.train.adv_clip_max = 1
    config.clip_eps = 0.0005
    config.train.llm_kl_beta = 0.0005
    config.gspo = True
    config.use_llm_lora = False

    # general config
    config.decay_type = 1
    config.mixed_precision = "fp16"
    config.save_freq = 40
    if "debug" in name:
        config.debug = True
    return config


def longlive_llm_diffusion_rl_mix():
    reward_fn = {
        "qwen3vl_local_score": 0.0,
        "qwen3vl_video_score": 0.85,
        "videopickscore_local_score": 0.0,
        "llm_judge_score": 0.15
    }
    config = _get_longlive_text_grpo_diffusion_flowgrpo_mix(
        base_model="longlive", n_gpus=32, gradient_step_per_epoch=2, dataset="temporal_order", reward_fn=reward_fn, name="longlive_llm_diffusion_rl_mix"
    )
    return config
