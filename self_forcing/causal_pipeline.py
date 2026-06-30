from typing import List, Optional
import torch
import json
from self_forcing.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from self_forcing.policy_model import MemoryPolicy
import torch.nn.functional as F
from self_forcing.demo_utils_memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation
from self_forcing.compute_log_prob_self_forcing import compute_log_prob
import json
import re
from contextlib import nullcontext

def get_autocast_context(accelerator=None, enabled=True):
    if not enabled:
        return nullcontext()

    if accelerator is not None:
        return accelerator.autocast()
    else:
        return nullcontext()

class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block


    def flowgrpo_inference_withllmpolicy_Diffusionmix_sample(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        return_log_prob: bool = False,
        gap_frame: int = 12,
        inner_diffusion_group: int = 1,
        inner_diffusion_frame_step: int = 15,
        policy_model: Optional[torch.nn.Module] = None,
        llm_semantic_policy: Optional[torch.nn.Module] = None,
        llm_policy_tokenizer: Optional[torch.nn.Module] = None,
        accelerator=None,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames

        if llm_semantic_policy is None:
            conditional_dict = self.text_encoder(
                text_prompts=text_prompts
            )
        # else:
        #     conditional_dict = {"prompt_embeds": prompt_embeds}


        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            kv_cache1 = self._initialize_kv_cache(
                batch_size=batch_size,
                num_blocks=num_blocks,
                dtype=noise.dtype,
                device=noise.device,
                return_c=True
            )
            crossattn_cache = self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device,
                return_c=True
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(kv_cache1)):
                kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=kv_cache1,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=kv_cache1,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        cache_decision = None
        rewritting_prompt = {}
        acc_conditional_dict = []
        if llm_semantic_policy is not None:
            # with torch.autocast(device_type="cuda", dtype=torch.float16):
            # llm_feature = self.llm_rewritting_prompt_withglobal(llm_semantic_policy, llm_policy_tokenizer, text_prompts, gap_frame=gap_frame, sample_frame=num_output_frames, return_log_prob=return_log_prob)

            llm_feature = self.llm_rewritting_prompt(llm_semantic_policy, llm_policy_tokenizer, text_prompts, gap_frame=gap_frame, sample_frame=num_output_frames, return_log_prob=return_log_prob) 
            promts_outputs = llm_feature["outputs"]
            prompt_for_reward = promts_outputs
            prompt_for_llm_rewards = llm_feature["prompt_for_llm_rewards"]
            format_reward = torch.tensor(llm_feature["format_reward"], device=noise.device, dtype=noise.dtype)
            llm_generation_ids = llm_feature["llm_generation_ids"].to(device=noise.device)
            llm_prompt_ids = llm_feature["llm_prompt_ids"].to(device=noise.device)
            respones_json_mask = llm_feature["respones_json_mask"].to(device=noise.device)
            # print(promts_outputs)
        # Step 3: Temporal denoising loop

        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames

        with get_autocast_context(accelerator=accelerator):
            for current_num_frames in all_num_frames:
                if current_start_frame >= inner_diffusion_frame_step:
                    break
                if profile:
                    block_start.record()

                noisy_input = noise[
                    :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
                
                if current_start_frame > 9:
                    cache_decision = torch.ones([batch_size, current_start_frame], device=noise.device, dtype=noise.dtype)
                    cache_decision[:, 1:-8] = 0

                # print(current_start_frame)
                if llm_semantic_policy is not None and current_start_frame % gap_frame == 0:
                    prompt_step_idx = current_start_frame // gap_frame
                    step_prompts = [promts_output[prompt_step_idx]["prompt"] for promts_output in promts_outputs]
                    # outputs = llm_semantic_policy.generation(message)
                    rewritting_prompt[f"num_frames_{current_start_frame}"] = step_prompts
                    conditional_dict = self.text_encoder(
                        text_prompts=step_prompts
                    )
                    acc_conditional_dict.append(conditional_dict["prompt_embeds"])

                    crossattn_cache = self._initialize_crossattn_cache(
                        batch_size=batch_size,
                        dtype=noise.dtype,
                        device=noise.device,
                        return_c=True
                    )

                # Step 3.1: Spatial denoising loop
                for index, current_timestep in enumerate(self.denoising_step_list):
                    # print(f"current_timestep: {current_timestep}")
                    # set current timestep
                    timestep = torch.ones(
                        [batch_size, current_num_frames],
                        device=noise.device,
                        dtype=torch.int64) * current_timestep

                    if index < len(self.denoising_step_list) - 1:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=kv_cache1,
                            crossattn_cache=crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                            cache_decision=cache_decision,
                            recording_cache=False
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                    else:
                        # for getting real output
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=kv_cache1,
                            crossattn_cache=crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                            cache_decision=cache_decision,
                            recording_cache=False
                        )

                # Step 3.2: record the model's output
                output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

                # Step 3.3: rerun with timestep zero to update KV cache using clean context
                context_timestep = torch.ones_like(timestep) * self.args.context_noise
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=kv_cache1,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_decision=cache_decision,
                )

                # Step 3.4: update the start and end frame indices
                current_start_frame += current_num_frames

            
            micro_batch_size = 1
            # import ipdb; ipdb.set_trace()
            group_kvcache = self.repeat_kv_cache1_for_group_sample(micro_batch_size, kv_cache1)
            if current_start_frame != 0:
                context_output = output[:, :current_start_frame].clone()
                conditional_dict["prompt_embeds"] = conditional_dict["prompt_embeds"].repeat(micro_batch_size, 1, 1)
            crossattn_cache = self._initialize_crossattn_cache(
                            batch_size=batch_size * micro_batch_size,
                            dtype=noise.dtype,
                            device=noise.device,
                            return_c=True
                        )

            del kv_cache1
            group_latents = []
            videos = []
            group_train_latents = []
            group_log_probs = []
            group_flow_preds = []
            for ind_g in range(0, inner_diffusion_group, micro_batch_size):
                cur_frame = 0

                # group_outputs = output
                output_group = torch.zeros_like(output)
                if current_start_frame != 0:
                    output_group[:, :current_start_frame] = context_output
                output_group = output_group.repeat(micro_batch_size, 1, 1, 1, 1)

                # ---- 2. rollout remaining frames ----
                for current_num_frames in all_num_frames:
                    if cur_frame < current_start_frame:
                        cur_frame += current_num_frames
                        continue
                    

                    noisy_input = noise[
                        :, cur_frame - num_input_frames:cur_frame + current_num_frames - num_input_frames
                    ]

                    # （可选）group-level stochasticity
                    if inner_diffusion_group > 1:
                        noisy_input = torch.randn_like(noisy_input.repeat(micro_batch_size, 1, 1, 1, 1))
                        # print(f"group index {ind_g} noisy_input stats: mean={noisy_input.mean():.4f}, std={noisy_input.std():.4f}")

                    if cur_frame == current_start_frame:
                        denoising_latents = [noisy_input]
                        log_probs = []
                        flow_preds = []

                    if cur_frame > 9:
                        cache_decision = torch.ones([batch_size * micro_batch_size, cur_frame], device=noise.device, dtype=noise.dtype)
                        cache_decision[:, 1:-8] = 0
                    else:
                        cache_decision = None



                    # ---- LLM prompt（保持一致 or 可改为group扰动）----
                    # import ipdb; ipdb.set_trace()
                    if llm_semantic_policy is not None and cur_frame % gap_frame == 0:
                        if ind_g == 0:
                            prompt_step_idx = cur_frame // gap_frame
                            step_prompts = [promts_output[prompt_step_idx]["prompt"] for promts_output in promts_outputs]

                            conditional_dict = self.text_encoder(text_prompts=step_prompts)
                            acc_conditional_dict.append(conditional_dict["prompt_embeds"])
                        else:
                            prompt_step_idx = cur_frame // gap_frame
                            conditional_dict = {
                                "prompt_embeds": prompt_embedings[:, :, :, prompt_step_idx]
                            }

                        conditional_dict["prompt_embeds"] = conditional_dict["prompt_embeds"].repeat(micro_batch_size, 1, 1)

                        crossattn_cache = self._initialize_crossattn_cache(
                            batch_size=batch_size * micro_batch_size,
                            dtype=noise.dtype,
                            device=noise.device,
                            return_c=True
                        )

                        # to do recache ?
                        
                    # ---- diffusion ----
                    for index, current_timestep in enumerate(self.denoising_step_list):

                        timestep = torch.ones(
                            [batch_size * micro_batch_size, current_num_frames],
                            device=noise.device,
                            dtype=torch.int64
                        ) * current_timestep

                        if index < len(self.denoising_step_list) - 1:
                            flow_pred, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=group_kvcache,
                                crossattn_cache=crossattn_cache,
                                current_start=cur_frame * self.frame_seq_length,
                                cache_decision=cache_decision,
                                recording_cache=False
                            )

                            next_timestep = self.denoising_step_list[index + 1]
                            sample_noise = torch.randn_like(denoised_pred)
                            next_noisy_input = (1 - next_timestep / 1000) * denoised_pred + (next_timestep / 1000) * sample_noise
                            log_prob = compute_log_prob(
                                noisy_input.float(), # x_t
                                next_noisy_input.float(), # x_t_1
                                flow_pred.float(), 
                                current_timestep.float() / 1000, 
                                next_timestep.float() / 1000
                            )
                            noisy_input = next_noisy_input
                            
                            if cur_frame == current_start_frame:
                                denoising_latents.append(noisy_input)
                                log_probs.append(log_prob)
                                flow_preds.append(flow_pred)


                        else:
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=group_kvcache,
                                crossattn_cache=crossattn_cache,
                                current_start=cur_frame * self.frame_seq_length,
                                cache_decision=cache_decision,
                                recording_cache=False
                            )

                    # ---- 写入 ----
                    if cur_frame == current_start_frame:
                        group_latents.append(denoised_pred.reshape(batch_size, micro_batch_size, *denoised_pred.shape[1:]))
                        group_train_latents.append(torch.stack(denoising_latents, dim=1).reshape(batch_size, micro_batch_size, len(denoising_latents), *denoising_latents[0].shape[1:]))
                        group_log_probs.append(torch.stack(log_probs, dim=1).reshape(batch_size, micro_batch_size, len(log_probs), *log_probs[0].shape[1:]))
                        group_flow_preds.append(torch.stack(flow_preds, dim=1).reshape(batch_size, micro_batch_size, len(flow_preds), *flow_preds[0].shape[1:]))

                    output_group[:, cur_frame:cur_frame + current_num_frames] = denoised_pred

                    # ---- update cache ----
                    context_timestep = torch.ones_like(timestep) * self.args.context_noise
                    self.generator(
                        noisy_image_or_video=denoised_pred,
                        conditional_dict=conditional_dict,
                        timestep=context_timestep,
                        kv_cache=group_kvcache,
                        crossattn_cache=crossattn_cache,
                        cache_decision=cache_decision,
                        current_start=cur_frame * self.frame_seq_length,
                    )

                    cur_frame += current_num_frames


                # Step 4: Decode the output
                for micro_index in range(micro_batch_size):
                    g_video = self.vae.decode_to_pixel(output_group[micro_index * batch_size:(micro_index + 1) * batch_size], use_cache=False)
                    g_video = (g_video * 0.5 + 0.5).clamp(0, 1)
                    videos.append(g_video)
                    self.vae.model.clear_cache()

                if ind_g == 0:
                    prompt_embedings = torch.stack(acc_conditional_dict, dim=-1)

        # del kv_cache1
        del crossattn_cache
        del group_kvcache
        # import ipdb; ipdb.set_trace()
        collate_data = {
            "video": torch.stack(videos, dim=1), # B, G, F, C, H, W
            "latent": output,
            # "train_latents": torch.cat(group_latents, dim=1), # B, G, 3, C, H, W
            "prompt_embedings": conditional_dict["prompt_embeds"],
            "diffusion_log_prob": torch.cat(group_log_probs, dim=1), # B, G, T, F, C, H, W
            "diffusion_train_latents": torch.cat(group_train_latents, dim=1), # B, G, T, F, C, H, W
            "diffusion_flow_preds": torch.cat(group_flow_preds, dim=1), # B, G, T, F, C, H, W
        }

        if llm_semantic_policy is not None:
            # prompt_embedings = torch.stack(acc_conditional_dict, dim=-1)
            collate_data["rewritting_prompt"] = rewritting_prompt
            collate_data["llm_generation_ids"] = llm_generation_ids
            collate_data["llm_prompt_ids"] = llm_prompt_ids
            collate_data["format_reward"] = format_reward
            collate_data["prompt_embedings"] = prompt_embedings
            collate_data["respones_json_mask"] = respones_json_mask
            collate_data["prompt_for_reward"] = prompt_for_reward
            collate_data["prompt_for_llm_rewards"] = prompt_for_llm_rewards
            if return_log_prob:
                collate_data["old_gen_log_probs"] = llm_feature["old_gen_log_probs"]
                # print("old_gen_log_probs dtype:", collate_data["old_gen_log_probs"].dtype)

        return collate_data 


    def repeat_kv_cache1_for_group_sample(self, micro_size, kv_cache1):
        group_kv_cache1 = []
        for index in range(self.num_transformer_blocks):
            group_kv_cache1.append({
                "k": kv_cache1[index]["k"].clone().repeat(micro_size, 1, 1, 1),
                "v": kv_cache1[index]["v"].clone().repeat(micro_size, 1, 1, 1),
                "global_end_index": kv_cache1[index]["global_end_index"].clone(),
                "local_end_index": kv_cache1[index]["local_end_index"].clone()
            })
        return group_kv_cache1

    def repeat_crossattn_cache_for_group_sample(self, micro_size, crossattn_cache):
        group_crossattn_cache = []
        for index in range(self.num_transformer_blocks):
            group_crossattn_cache.append({
                "k": crossattn_cache[index]["k"].clone().repeat(micro_size, 1, 1, 1),
                "v": crossattn_cache[index]["v"].clone().repeat(micro_size, 1, 1, 1),
                "is_init": crossattn_cache[index]["is_init"]
            })
        return group_crossattn_cache

    def _initialize_kv_cache(self, batch_size, num_blocks, dtype, device, return_c=False):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 4680 * num_blocks
            if self.independent_first_frame:
                kv_cache_size = kv_cache_size + 1560

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        if return_c:
            return kv_cache1
        else:
            self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device, return_c=False):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        if return_c:
            return crossattn_cache
        else:
            self.crossattn_cache = crossattn_cache  # always store the clean cache 

    def llm_rewritting_prompt(self, llm_policy, llm_policy_tokenizer, usr_prompts, gap_frame, sample_frame, return_log_prob=False):
        PE_detail_instruction_prompt = """
You are an expert in temporal reasoning and cinematic prompt enhancement for video generation models.

Think Content Rules:
1. Limit think content to at most 256 tokens.
2. Use concise reasoning only when necessary for temporal planning.
3. Avoid repetition and overly verbose explanations.

A video diffusion model generates videos progressively over time.
Given the original prompt describing the entire video, split the video evenly into {total_steps} sequential temporal steps
and rewrite the prompt for each step into a high-quality cinematic video generation prompt.

The rewritten prompts should function as visually rich video captions for each temporal segment, rather than short action summaries.

The output should optimize for:
- realistic motion depiction
- smooth temporal progression
- visually coherent scene evolution
- cinematic atmosphere
- detailed texture and material perception
- physically plausible behavior
- high-quality video generation
- autoregressive temporal consistency

Step Prompt Rules:
1. Each step should describe only what is visually happening during that segment of the video.
2. Write each step as a rich cinematic video caption, NOT as a short event label.
3. Focus on how motion visually evolves, not only what action occurs.
4. Maintain logical and chronological consistency with the original prompt.
5. Use present tense.
6. Preserve consistency of subjects, objects, lighting, and environment across steps.
7. Do NOT describe past or future events outside the current step.
8. Do NOT introduce new characters, objects, or unrelated events.
9. Describe natural motion details when relevant, including:
   - movement trajectory
   - drifting, rotation, swaying, tilting
   - gradual acceleration or slowing
   - object interaction and physical response
10. Enhance prompts with cinematic and visually immersive details when appropriate, including:
   - natural lighting and shadows
   - atmospheric depth
   - soft wind, fog, dust, particles, reflections, or ambient motion
   - detailed textures and material properties
   - subtle environmental responses
   - spatial depth and scene composition
11. You MAY enrich prompts with high-quality visual descriptors such as:
   - cinematic
   - realistic
   - highly detailed
   - soft natural lighting
   - shallow depth of field
   - volumetric lighting
   - film-like atmosphere
   - realistic texture
   - smooth motion
   but only when they naturally fit the scene.
12. Intermediate steps should capture partial motion evolution rather than only start/end states.
13. Prefer continuous motion progression over abrupt state transitions.
14. Avoid repetitive wording, excessive style tags, or overloaded aesthetic keywords.
15. Keep prompts concise but visually dense and informative for video generation.
16. Each step should feel like a coherent segment from a high-quality cinematic video.

Output format:
Return a JSON array with {total_steps} elements.

Each element contains:
- "step_index": step index (0-based)
- "prompt": a detailed rewritten prompt for this step

Input:
Original prompt:
{original_prompt}
"""

        PE_global_instruction_prompt = """
You are an expert in rewriting prompts for video generation models.

Goal:
Rewrite the original prompt to improve its suitability for video diffusion models,
with a focus on temporal coherence, motion consistency, physical plausibility,
and visually rich scene evolution.

Think Content Rules:

1. Limit reasoning to at most 256 tokens.
2. Avoid repetition.
3. Focus only on necessary transformations.

Optimization Objectives:

1. Temporal Coherence

* Ensure actions are described in a clear and logical temporal order.
* Make implicit temporal relationships explicit when beneficial.
* Preserve continuous progression of object states throughout the video.
* Encourage smooth evolution rather than disconnected events.

2. Motion Consistency

* Maintain a consistent subject identity and motion trajectory.
* Describe motion as a continuous process rather than isolated states.
* When applicable, clarify movement direction, speed changes, rotation,
  deformation, interaction responses, or gradual transitions.
* Avoid abrupt, discontinuous, or unrealistic motion changes.

3. Physical Plausibility

* Ensure actions and interactions follow basic physical intuition.
* Preserve realistic object behavior, contact dynamics, and environmental responses.
* Avoid contradictory or impossible descriptions.

4. Scene Evolution Awareness

* Preserve consistency of environment, lighting, objects, and scene layout.
* Emphasize how the scene naturally evolves over time.
* Maintain coherent relationships between subjects and surroundings.
* Avoid introducing unexplained state changes.

5. Visual Quality Enhancement

* Rewrite the prompt as a high-quality cinematic video description.
* Enhance visual richness when appropriate through:

  * natural lighting and shadows
  * realistic textures and material properties
  * atmospheric depth
  * reflections, particles, dust, fog, or subtle ambient motion
  * spatial composition and depth cues
* Focus on details that improve video generation quality while preserving semantics.

Diversity Requirements:

1. Vary sentence structure and phrasing.
2. Adjust emphasis between motion, appearance, and environment when helpful.
3. Slightly enrich visual details only when consistent with the original prompt.
4. Do NOT introduce new objects, characters, actions, or events.
5. Avoid rigid templates and excessive aesthetic keywords.

Global Prompt Rules:

1. Preserve the original semantics.
2. Use present tense.
3. Do NOT add new entities or actions.
4. Improve clarity for video generation.
5. Describe the overall motion progression and scene evolution when applicable.
6. Preserve subject, object, lighting, and environmental consistency.
7. Prefer continuous motion and gradual transitions over discrete state descriptions.
8. Keep the prompt concise but visually rich.
9. The rewritten prompt should read like a high-quality cinematic video caption
   describing the complete video.

Output format:
Return only the final rewritten global prompt.

Input:
Original prompt:
{original_prompt}
"""

        texts = []
        if sample_frame != gap_frame:
            total_steps = sample_frame // gap_frame

            if sample_frame % gap_frame != 0:
                total_steps += 1
            instruction_prompt = PE_detail_instruction_prompt
        else:
            total_steps = 1
            instruction_prompt = PE_global_instruction_prompt


        for usr_prompt in usr_prompts:
            messages = [{
                "role": "user",
                "content": instruction_prompt.format(
                    original_prompt=usr_prompt,
                    total_steps=str(total_steps)
                )
            }]

            text = llm_policy_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )

            texts.append(text)

        model_inputs = llm_policy_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True).to(llm_policy.device)

        # conduct text completion
        # LLM 以 fp16 运行，用显式的 fp16 autocast 覆盖外层可能存在的 bf16 上下文
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            generated_ids = llm_policy.generate(
                **model_inputs,
                max_new_tokens=2024,
                do_sample=True,
                temperature=1.0,
                top_p=0.7,
                repetition_penalty=1.0,
                return_dict_in_generate=True,
                output_scores=True,       # 生成 logits
            )

        llm_generation_ids = generated_ids.sequences[:, model_inputs.input_ids.shape[1]:]
        llm_prompt_ids = model_inputs.input_ids

        respones_json_mask = (generated_ids.sequences != llm_policy_tokenizer.pad_token_id).float()

        outputs = []
        format_reward = []
        prompt_for_llm_rewards = []
        for i in range(len(texts)):

            sequence = generated_ids.sequences[i][len(model_inputs.input_ids[i]):]  # [B, L]

            output_ids = sequence.tolist()

            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            thinking_content = llm_policy_tokenizer.decode(
                output_ids[:index], skip_special_tokens=True
            ).strip("\n")

            respones_json_mask[i, :index] = 0.0

            content = llm_policy_tokenizer.decode(
                output_ids[index:], skip_special_tokens=True
            ).strip("\n")

            try:
                if total_steps == 1:
                    step_prompts = [{"step_index": 0, "prompt": content}]
                    if index == 0:
                        format_reward.append(0.0)
                    else:
                        format_reward.append(1.0)
                else:
                    step_prompts = json.loads(content)
                    generated_steps = len(step_prompts)
                    if generated_steps < total_steps:
                        last_prompt = step_prompts[-1]["prompt"]
                        for i in range(generated_steps, total_steps):
                            step_prompts.append({
                                "step_index": i,
                                "prompt": last_prompt
                            })
                        format_reward.append(0.5)
                    else:
                        for prompt_step_idx in range(total_steps):
                            step_prompt = step_prompts[prompt_step_idx]["prompt"]
                        format_reward.append(1.0)

                    prompt_for_llm_rewards.append(step_prompts)
                    for prompt_step_idx in range(generated_steps - 1):
                        step_prompts[prompt_step_idx]["prompt"] = step_prompts[prompt_step_idx]["prompt"] + step_prompts[prompt_step_idx + 1]["prompt"]
            
                print("step_prompts:", step_prompts)
                
            except:
                step_prompts = [{"step_index": step, "prompt": usr_prompts[i]} for step in range(total_steps)]
                format_reward.append(0.0)
                prompt_for_llm_rewards.append(step_prompts)
            outputs.append(step_prompts)
        
        respones_json_mask = respones_json_mask[:, model_inputs.input_ids.shape[1]:]
        llm_feature = {
            "llm_prompt_ids": model_inputs.input_ids,
            "llm_generation_ids": llm_generation_ids,
            "outputs": outputs,
            "prompt_for_llm_rewards": prompt_for_llm_rewards,
            "format_reward": format_reward,
            "respones_json_mask": respones_json_mask
        }

        if return_log_prob:
            with torch.no_grad():
                attention_mask = (generated_ids.sequences != llm_policy_tokenizer.pad_token_id).float()
                # LLM logprob 计算也在 fp16 下进行
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    old_outputs = llm_policy(input_ids=generated_ids.sequences, attention_mask=attention_mask)
                labels = generated_ids.sequences[:, llm_prompt_ids.shape[1]:]

                old_logits = old_outputs.logits.float()
                old_logits = old_logits[:, llm_prompt_ids.shape[1] - 1:-1]
                old_logits = old_logits / 1.0
                old_logits = F.log_softmax(old_logits, dim=-1)

                old_gen_log_probs = old_logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                llm_feature["old_gen_log_probs"] = old_gen_log_probs

        return llm_feature

