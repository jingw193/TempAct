# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional
import torch

from self_forcing.longlive_wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from self_forcing.demo_utils_memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation
from self_forcing.causal_pipeline_longlive import CausalInferencePipeline
import torch.distributed as dist
from self_forcing.debug_option import DEBUG
from self_forcing.compute_log_prob_self_forcing import llm_rewritting_prompt, compute_log_prob
from contextlib import nullcontext

def get_autocast_context(accelerator=None, enabled=True):
    if not enabled:
        return nullcontext()

    if accelerator is not None:
        return accelerator.autocast()
    else:
        return nullcontext()

class InteractiveCausalInferencePipeline(CausalInferencePipeline):
    def __init__(
        self,
        args,
        device,
        *,
        generator: WanDiffusionWrapper | None = None,
        text_encoder: WanTextEncoder | None = None,
        vae: WanVAEWrapper | None = None,
    ):
        super().__init__(args, device, generator=generator, text_encoder=text_encoder, vae=vae)
        self.global_sink = getattr(args, "global_sink", False)

    # Internal helpers
    def _recache_after_switch(self, output, current_start_frame, new_conditional_dict):
        if not self.global_sink:
            # reset kv cache
            for block_idx in range(self.num_transformer_blocks):
                cache = self.kv_cache1[block_idx]
                cache["k"].zero_()
                cache["v"].zero_()
                # cache["global_end_index"].zero_()
                # cache["local_end_index"].zero_()
            
        # reset cross-attention cache
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

        # recache
        if current_start_frame == 0:
            return
        
        num_recache_frames = current_start_frame if self.local_attn_size == -1 else min(self.local_attn_size, current_start_frame)
        recache_start_frame = current_start_frame - num_recache_frames
        
        frames_to_recache = output[:, recache_start_frame:current_start_frame]
        # move to gpu
        if frames_to_recache.device.type == 'cpu':
            target_device = next(self.generator.parameters()).device
            frames_to_recache = frames_to_recache.to(target_device)
        batch_size = frames_to_recache.shape[0]
        print(f"num_recache_frames: {num_recache_frames}, recache_start_frame: {recache_start_frame}, current_start_frame: {current_start_frame}")
        
        # prepare blockwise causal mask
        device = frames_to_recache.device
        block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
            device=device,
            num_frames=num_recache_frames,
            frame_seqlen=self.frame_seq_length,
            num_frame_per_block=self.num_frame_per_block,
            local_attn_size=self.local_attn_size
        )
        
        context_timestep = torch.ones([batch_size, num_recache_frames], 
                                    device=device, dtype=torch.int64) * self.args.context_noise
        
        self.generator.model.block_mask = block_mask
        
        # recache
        with torch.no_grad():
            self.generator(
                noisy_image_or_video=frames_to_recache,
                conditional_dict=new_conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=recache_start_frame * self.frame_seq_length,
                sink_recache_after_switch=not self.global_sink,
            )
        
        # reset cross-attention cache
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

    def _recache_after_switch_input_cache(self, output, kv_cache1, crossattn_cache, current_start_frame, new_conditional_dict):
        if not self.global_sink:
            # reset kv cache
            for block_idx in range(self.num_transformer_blocks):
                cache = kv_cache1[block_idx]
                cache["k"].zero_()
                cache["v"].zero_()
                # cache["global_end_index"].zero_()
                # cache["local_end_index"].zero_()
            
        # reset cross-attention cache
        for blk in crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

        # recache
        if current_start_frame == 0:
            return
        
        num_recache_frames = current_start_frame if self.local_attn_size == -1 else min(self.local_attn_size, current_start_frame)
        recache_start_frame = current_start_frame - num_recache_frames
        
        frames_to_recache = output[:, recache_start_frame:current_start_frame]
        # move to gpu
        if frames_to_recache.device.type == 'cpu':
            target_device = next(self.generator.parameters()).device
            frames_to_recache = frames_to_recache.to(target_device)
        batch_size = frames_to_recache.shape[0]

        if DEBUG:
            print(f"num_recache_frames: {num_recache_frames}, recache_start_frame: {recache_start_frame}, current_start_frame: {current_start_frame}")
        
        # prepare blockwise causal mask
        device = frames_to_recache.device
        block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
            device=device,
            num_frames=num_recache_frames,
            frame_seqlen=self.frame_seq_length,
            num_frame_per_block=self.num_frame_per_block,
            local_attn_size=self.local_attn_size
        )
        
        context_timestep = torch.ones([batch_size, num_recache_frames], 
                                    device=device, dtype=torch.int64) * self.args.context_noise
        
        self.generator.model.block_mask = block_mask
        
        # recache
        with torch.no_grad():
            self.generator(
                noisy_image_or_video=frames_to_recache,
                conditional_dict=new_conditional_dict,
                timestep=context_timestep,
                kv_cache=kv_cache1,
                crossattn_cache=crossattn_cache,
                current_start=recache_start_frame * self.frame_seq_length,
                sink_recache_after_switch=not self.global_sink,
            )
        
        # reset cross-attention cache
        for blk in crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

        return kv_cache1, crossattn_cache

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

    def inference(
        self,
        noise: torch.Tensor,
        *,
        text_prompts_list: List[List[str]],
        switch_frame_indices: List[int],
        return_latents: bool = False,
        low_memory: bool = False,
    ):
        """Generate a video and switch prompts at specified frame indices.

        Args:
            noise: Noise tensor, shape = (B, T_out, C, H, W).
            text_prompts_list: List[List[str]], length = N_seg. Prompt list used for segment i (aligned with batch).
            switch_frame_indices: List[int], length = N_seg - 1. The i-th value indicates that when generation reaches this frame (inclusive)
                we start using the prompts for segment i+1.
            return_latents: Whether to also return the latent tensor.
            low_memory: Enable low-memory mode.
        """
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert len(text_prompts_list) >= 1, "text_prompts_list must not be empty"
        assert len(switch_frame_indices) == len(text_prompts_list) - 1, (
            "length of switch_frame_indices should be one less than text_prompts_list"
        )
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        
        # encode all prompts
        print(text_prompts_list)
        cond_list = [self.text_encoder(text_prompts=p) for p in text_prompts_list]

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder,
                target_device=gpu,
                preserved_memory_gb=gpu_memory_preservation,
            )

        output_device = torch.device('cpu') if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype
        )

        # initialize caches
        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        kv_policy = ""
        if local_attn_cfg != -1:
            # local attention
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local, size={local_attn_cfg}"
        else:
            # global attention
            kv_cache_size = num_output_frames * self.frame_seq_length
            kv_policy = "global (-1)"
        print(f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})")

        self._initialize_kv_cache(
            batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_cache_size_override=kv_cache_size
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device
        )

        current_start_frame = 0
        self.generator.model.local_attn_size = self.local_attn_size
        print(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size)

        # temporal denoising by blocks
        all_num_frames = [self.num_frame_per_block] * num_blocks
        segment_idx = 0  # current segment index
        next_switch_pos = (
            switch_frame_indices[segment_idx]
            if segment_idx < len(switch_frame_indices)
            else None
        )

        if DEBUG:
            print("[MultipleSwitch] all_num_frames", all_num_frames)
            print("[MultipleSwitch] switch_frame_indices", switch_frame_indices)

        for current_num_frames in all_num_frames:
            if next_switch_pos is not None and current_start_frame >= next_switch_pos:
                segment_idx += 1
                self._recache_after_switch(output, current_start_frame, cond_list[segment_idx])
                if DEBUG:
                    print(
                        f"[MultipleSwitch] Switch to segment {segment_idx} at frame {current_start_frame}"
                    )
                next_switch_pos = (
                    switch_frame_indices[segment_idx]
                    if segment_idx < len(switch_frame_indices)
                    else None
                )
                print(f"segment_idx: {segment_idx}")
                print(f"text_prompts_list[segment_idx]: {text_prompts_list[segment_idx]}")
            cond_in_use = cond_list[segment_idx]

            noisy_input = noise[
                :, current_start_frame : current_start_frame + current_num_frames
            ]

            # ---------------- Spatial denoising loop ----------------
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = (
                    torch.ones([batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64)
                    * current_timestep
                )

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_in_use,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep
                        * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long
                        ),
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_in_use,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )

            # Record output
            output[:, current_start_frame : current_start_frame + current_num_frames] = denoised_pred.to(output.device)

            # rerun with clean context to update cache
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=cond_in_use,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            # Update frame pointer
            current_start_frame += current_num_frames

        # Standard decoding
        video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, output
        return video 

    def grpo_llm_inference(
        self,
        noise: torch.Tensor,
        *,
        text_prompts_list: List[str],
        return_latents: bool = False,
        low_memory: bool = False,
        return_log_prob: bool = False,
        gap_frame: int = 12,
        llm_semantic_policy: Optional[torch.nn.Module] = None,
        llm_policy_tokenizer: Optional[torch.nn.Module] = None,
        accelerator=None,
    ):
        """Generate a video and switch prompts at specified frame indices.

        Args:
            noise: Noise tensor, shape = (B, T_out, C, H, W).
            text_prompts_list: List[List[str]], length = N_seg. Prompt list used for segment i (aligned with batch).
            switch_frame_indices: List[int], length = N_seg - 1. The i-th value indicates that when generation reaches this frame (inclusive)
                we start using the prompts for segment i+1.
            return_latents: Whether to also return the latent tensor.
            low_memory: Enable low-memory mode.
        """
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        # encode all prompts
        if llm_semantic_policy is not None:
            llm_feature = llm_rewritting_prompt(llm_semantic_policy, llm_policy_tokenizer, text_prompts_list, gap_frame=gap_frame, sample_frame=num_output_frames, return_log_prob=return_log_prob) 
            promts_outputs = llm_feature["outputs"] # list[list[dict{"step_index": int, "prompt": str}]]
            prompt_for_reward = promts_outputs
            prompt_for_llm_rewards = llm_feature["prompt_for_llm_rewards"]
            format_reward = torch.tensor(llm_feature["format_reward"], device=noise.device, dtype=noise.dtype)
            llm_generation_ids = llm_feature["llm_generation_ids"].to(device=noise.device)
            llm_prompt_ids = llm_feature["llm_prompt_ids"].to(device=noise.device)
            respones_json_mask = llm_feature["respones_json_mask"].to(device=noise.device)

            switch_frame_indices = [gap_frame * (i + 1) for i in range(len(promts_outputs[0]) - 1)]

            step_prompts_list = []
            for step_idx in range(len(promts_outputs[0])):
                step_prompts_list.append([p[step_idx]["prompt"] for p in promts_outputs])

            cond_list = [self.text_encoder(text_prompts=p) for p in step_prompts_list]
        else:
            cond_list = [self.text_encoder(text_prompts=p) for p in text_prompts_list]
            switch_frame_indices = [gap_frame * (i + 1) for i in range(len(text_prompts_list[0]) - 1)]

        if DEBUG:
            print(f"switch_frame_indices: {switch_frame_indices}")

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # initialize caches
        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        kv_policy = ""
        if local_attn_cfg != -1:
            # local attention
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local, size={local_attn_cfg}"
        else:
            # global attention
            kv_cache_size = num_output_frames * self.frame_seq_length
            kv_policy = "global (-1)"
        if DEBUG:
            print(f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})")

        kv_cache1 = self._initialize_kv_cache(
            batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_cache_size_override=kv_cache_size,
            return_cache=True
        )
        crossattn_cache = self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device,
            return_cache=True
        )

        current_start_frame = 0
        self.generator.model.local_attn_size = self.local_attn_size
        if DEBUG:
            print(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size)

        # temporal denoising by blocks
        all_num_frames = [self.num_frame_per_block] * num_blocks
        segment_idx = 0  # current segment index
        next_switch_pos = (
            switch_frame_indices[segment_idx]
            if segment_idx < len(switch_frame_indices)
            else None
        )

        if DEBUG:
            print("[MultipleSwitch] all_num_frames", all_num_frames)
            print("[MultipleSwitch] switch_frame_indices", switch_frame_indices)

        with get_autocast_context(accelerator=accelerator):
            for current_num_frames in all_num_frames:
                if next_switch_pos is not None and current_start_frame >= next_switch_pos:
                    segment_idx += 1
                    kv_cache1, crossattn_cache = self._recache_after_switch_input_cache(output, kv_cache1, crossattn_cache, current_start_frame, cond_list[segment_idx])
                    if DEBUG:
                        print(
                            f"[MultipleSwitch] Switch to segment {segment_idx} at frame {current_start_frame}"
                        )
                    next_switch_pos = (
                        switch_frame_indices[segment_idx]
                        if segment_idx < len(switch_frame_indices)
                        else None
                    )
                    if DEBUG:
                        print(f"segment_idx: {segment_idx}")
                        if llm_semantic_policy is not None:
                            print(f"step_prompts_list[segment_idx]: {step_prompts_list[segment_idx]}")
                        else:   
                            print(f"text_prompts_list[segment_idx]: {text_prompts_list[segment_idx]}")

                cond_in_use = cond_list[segment_idx]

                noisy_input = noise[
                    :, current_start_frame : current_start_frame + current_num_frames
                ]

                # ---------------- Spatial denoising loop ----------------
                for index, current_timestep in enumerate(self.denoising_step_list):
                    timestep = (
                        torch.ones([batch_size, current_num_frames],
                        device=noise.device,
                        dtype=torch.int64)
                        * current_timestep
                    )

                    if index < len(self.denoising_step_list) - 1:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=cond_in_use,
                            timestep=timestep,
                            kv_cache=kv_cache1,
                            crossattn_cache=crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep
                            * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long
                            ),
                        ).unflatten(0, denoised_pred.shape[:2])
                    else:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=cond_in_use,
                            timestep=timestep,
                            kv_cache=kv_cache1,
                            crossattn_cache=crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                        )

                # Record output
                output[:, current_start_frame : current_start_frame + current_num_frames] = denoised_pred.to(output.device)

                # rerun with clean context to update cache
                context_timestep = torch.ones_like(timestep) * self.args.context_noise
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=cond_in_use,
                    timestep=context_timestep,
                    kv_cache=kv_cache1,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )

                # Update frame pointer
                current_start_frame += current_num_frames

        # Standard decoding
        video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        prompt_embedings = torch.stack([cond["prompt_embeds"] for cond in cond_list], dim=-1)

        del crossattn_cache
        del kv_cache1
        # import ipdb; ipdb.set_trace()
        collate_data = {
            "video": video,
            "latent": output,
            "prompt_embedings": prompt_embedings,
        }

        if llm_semantic_policy is not None:
            collate_data["llm_generation_ids"] = llm_generation_ids
            collate_data["llm_prompt_ids"] = llm_prompt_ids
            collate_data["format_reward"] = format_reward
            collate_data["respones_json_mask"] = respones_json_mask
            collate_data["prompt_for_reward"] = prompt_for_reward
            collate_data["prompt_for_llm_reward"] = prompt_for_llm_rewards
            if return_log_prob:
                collate_data["old_gen_log_probs"] = llm_feature["old_gen_log_probs"]
                # print("old_gen_log_probs dtype:", collate_data["old_gen_log_probs"].dtype)

        return collate_data 
    
    def flowgrpo_llm_diffusion_mix_inference(
        self,
        noise: torch.Tensor,
        *,
        text_prompts_list: List[str],
        return_latents: bool = False,
        low_memory: bool = False,
        return_log_prob: bool = False,
        gap_frame: int = 12,
        inner_diffusion_group: int = 1,
        inner_diffusion_frame_step: int = 15,
        llm_semantic_policy: Optional[torch.nn.Module] = None,
        llm_policy_tokenizer: Optional[torch.nn.Module] = None,
        accelerator=None,
    ):
        """Generate a video and switch prompts at specified frame indices.

        Args:
            noise: Noise tensor, shape = (B, T_out, C, H, W).
            text_prompts_list: List[List[str]], length = N_seg. Prompt list used for segment i (aligned with batch).
            switch_frame_indices: List[int], length = N_seg - 1. The i-th value indicates that when generation reaches this frame (inclusive)
                we start using the prompts for segment i+1.
            return_latents: Whether to also return the latent tensor.
            low_memory: Enable low-memory mode.
        """
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        # encode all prompts
        llm_feature = llm_rewritting_prompt(llm_semantic_policy, llm_policy_tokenizer, text_prompts_list, gap_frame=gap_frame, sample_frame=num_output_frames, return_log_prob=return_log_prob) 
        promts_outputs = llm_feature["outputs"] # list[list[dict{"step_index": int, "prompt": str}]]
        prompt_for_reward = promts_outputs
        prompt_for_llm_rewards = llm_feature["prompt_for_llm_rewards"]
        format_reward = torch.tensor(llm_feature["format_reward"], device=noise.device, dtype=noise.dtype)
        llm_generation_ids = llm_feature["llm_generation_ids"].to(device=noise.device)
        llm_prompt_ids = llm_feature["llm_prompt_ids"].to(device=noise.device)
        respones_json_mask = llm_feature["respones_json_mask"].to(device=noise.device)

        switch_frame_indices = [gap_frame * (i + 1) for i in range(len(promts_outputs[0]) - 1)]
        # print(f"switch_frame_indices: {switch_frame_indices}")

        step_prompts_list = []
        for step_idx in range(len(promts_outputs[0])):
            step_prompts_list.append([p[step_idx]["prompt"] for p in promts_outputs])

        cond_list = [self.text_encoder(text_prompts=p) for p in step_prompts_list]

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # initialize caches
        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        kv_policy = ""
        if local_attn_cfg != -1:
            # local attention
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local, size={local_attn_cfg}"
        else:
            # global attention
            kv_cache_size = num_output_frames * self.frame_seq_length
            kv_policy = "global (-1)"
        
        if DEBUG:
            print(f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})")

        kv_cache1 = self._initialize_kv_cache(
            batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_cache_size_override=kv_cache_size,
            return_cache=True
        )
        crossattn_cache = self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device,
            return_cache=True
        )

        current_start_frame = 0
        self.generator.model.local_attn_size = self.local_attn_size
        # 
        self._set_all_modules_max_attention_size(self.local_attn_size)

        # temporal denoising by blocks
        all_num_frames = [self.num_frame_per_block] * num_blocks
        segment_idx = 0  # current segment index
        next_switch_pos = (
            switch_frame_indices[segment_idx]
            if segment_idx < len(switch_frame_indices)
            else None
        )

        if DEBUG:
            print(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
            print("[MultipleSwitch] all_num_frames", all_num_frames)
            print("[MultipleSwitch] switch_frame_indices", switch_frame_indices)

        rewritting_prompt = {}

        with get_autocast_context(accelerator=accelerator):
            # ================================================================
            # Phase 1: generate shared prefix frames up to inner_diffusion_frame_step
            # ================================================================
            for current_num_frames in all_num_frames:
                # Stop Phase 1 once enough prefix frames have been generated
                if current_start_frame >= inner_diffusion_frame_step:
                    break

                # Check for prompt switch before generating this block
                if next_switch_pos is not None and current_start_frame >= next_switch_pos:
                    segment_idx += 1
                    kv_cache1, crossattn_cache = self._recache_after_switch_input_cache(
                        output, kv_cache1, crossattn_cache, current_start_frame, cond_list[segment_idx]
                    )
                    if DEBUG:
                        print(
                            f"[MultipleSwitch] Switch to segment {segment_idx} at frame {current_start_frame}"
                        )
                    next_switch_pos = (
                        switch_frame_indices[segment_idx]
                        if segment_idx < len(switch_frame_indices)
                        else None
                    )
                    if DEBUG:
                        print(f"segment_idx: {segment_idx}")
                        print(f"step_prompts_list[segment_idx]: {step_prompts_list[segment_idx]}")

                cond_in_use = cond_list[segment_idx]

                # Track rewriting prompts at each gap_frame boundary
                if current_start_frame % gap_frame == 0:
                    prompt_step_idx = current_start_frame // gap_frame
                    if prompt_step_idx < len(step_prompts_list):
                        rewritting_prompt[f"num_frames_{current_start_frame}"] = step_prompts_list[prompt_step_idx]

                noisy_input = noise[
                    :, current_start_frame : current_start_frame + current_num_frames
                ]

                # ---------------- Spatial denoising loop ----------------
                for index, current_timestep in enumerate(self.denoising_step_list):
                    timestep = (
                        torch.ones(
                            [batch_size, current_num_frames],
                            device=noise.device,
                            dtype=torch.int64,
                        )
                        * current_timestep
                    )

                    if index < len(self.denoising_step_list) - 1:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=cond_in_use,
                            timestep=timestep,
                            kv_cache=kv_cache1,
                            crossattn_cache=crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep
                            * torch.ones(
                                [batch_size * current_num_frames],
                                device=noise.device,
                                dtype=torch.long,
                            ),
                        ).unflatten(0, denoised_pred.shape[:2])
                    else:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=cond_in_use,
                            timestep=timestep,
                            kv_cache=kv_cache1,
                            crossattn_cache=crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                        )

                # Record output
                output[:, current_start_frame : current_start_frame + current_num_frames] = denoised_pred.to(output.device)

                # Rerun with clean context to update KV cache
                context_timestep = torch.ones_like(timestep) * self.args.context_noise
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=cond_in_use,
                    timestep=context_timestep,
                    kv_cache=kv_cache1,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )

                current_start_frame += current_num_frames

            # ================================================================
            # Phase 2: group sampling from current_start_frame onwards
            # ================================================================
            micro_batch_size = 1
            context_output = output[:, :current_start_frame].clone() if current_start_frame > 0 else None
            final_segment_idx = segment_idx  # segment index at the end of Phase 1

            # Clone Phase 1 KV cache as the shared base for all groups, then free Phase 1 caches
            
            # del kv_cache1
            del crossattn_cache

            # Fresh cross-attention cache for group sampling
            group_crossattn_cache = self._initialize_crossattn_cache(
                batch_size=batch_size * micro_batch_size,
                dtype=noise.dtype,
                device=noise.device,
                return_cache=True,
            )

            group_latents = []
            videos = []
            group_train_latents = []
            group_log_probs = []
            group_flow_preds = []

            for ind_g in range(0, inner_diffusion_group, micro_batch_size):
                group_kvcache = self.repeat_kv_cache1_for_group_sample(micro_batch_size, kv_cache1)
                cur_frame = 0

                # Reset output buffer; prefill with Phase-1 prefix frames
                output_group = torch.zeros_like(output)
                if current_start_frame > 0 and context_output is not None:
                    output_group[:, :current_start_frame] = context_output
                output_group = output_group.repeat(micro_batch_size, 1, 1, 1, 1)

                # Reset segment tracking to where Phase 1 ended
                seg_idx_g = final_segment_idx
                # Next prompt switch at or after current_start_frame
                next_switch_pos_g = (
                    switch_frame_indices[seg_idx_g]
                    if seg_idx_g < len(switch_frame_indices)
                    else None
                )

                for current_num_frames in all_num_frames:
                    # Skip prefix frames already generated in Phase 1
                    if cur_frame < current_start_frame:
                        cur_frame += current_num_frames
                        continue

                    # ---- Prompt switch: recache KV cache under new segment prompt ----
                    if next_switch_pos_g is not None and cur_frame >= next_switch_pos_g:
                        seg_idx_g += 1
                        group_kvcache, group_crossattn_cache = self._recache_after_switch_input_cache(
                            output_group,
                            group_kvcache,
                            group_crossattn_cache,
                            cur_frame,
                            cond_list[seg_idx_g],
                        )
                        next_switch_pos_g = (
                            switch_frame_indices[seg_idx_g]
                            if seg_idx_g < len(switch_frame_indices)
                            else None
                        )
                        # Track rewriting prompts (first group only to avoid duplicates)
                        if ind_g == 0:
                            rewritting_prompt[f"num_frames_{cur_frame}"] = step_prompts_list[seg_idx_g]

                    # Conditional embedding for current segment (broadcast to micro_batch)
                    cond_in_use = {
                        "prompt_embeds": cond_list[seg_idx_g]["prompt_embeds"].repeat(micro_batch_size, 1, 1)
                    }

                    # Fresh noise for group diversity (re-sample for each group)
                    noisy_input = noise[:, cur_frame : cur_frame + current_num_frames]
                    if inner_diffusion_group > 1:
                        noisy_input = torch.randn_like(
                            noisy_input.repeat(micro_batch_size, 1, 1, 1, 1)
                        )

                    # Initialise per-trajectory bookkeeping at the first newly-generated frame
                    if cur_frame == current_start_frame:
                        denoising_latents = [noisy_input]
                        log_probs = []
                        flow_preds = []

                    # ---- Spatial denoising with flow prediction ----
                    for index, current_timestep in enumerate(self.denoising_step_list):
                        timestep = (
                            torch.ones(
                                [batch_size * micro_batch_size, current_num_frames],
                                device=noise.device,
                                dtype=torch.int64,
                            )
                            * current_timestep
                        )

                        if index < len(self.denoising_step_list) - 1:
                            flow_pred, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=cond_in_use,
                                timestep=timestep,
                                kv_cache=group_kvcache,
                                crossattn_cache=group_crossattn_cache,
                                current_start=cur_frame * self.frame_seq_length
                            )
                            next_timestep = self.denoising_step_list[index + 1]
                            sample_noise = torch.randn_like(denoised_pred)
                            next_noisy_input = (
                                (1 - next_timestep / 1000) * denoised_pred
                                + (next_timestep / 1000) * sample_noise
                            )
                            log_prob = compute_log_prob(
                                noisy_input.float(),
                                next_noisy_input.float(),
                                flow_pred.float(),
                                current_timestep.float() / 1000,
                                next_timestep.float() / 1000,
                            )
                            noisy_input = next_noisy_input

                            # Only record trajectory data for the first newly-generated frame
                            if cur_frame == current_start_frame:
                                denoising_latents.append(noisy_input)
                                log_probs.append(log_prob)
                                flow_preds.append(flow_pred)

                        else:
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=cond_in_use,
                                timestep=timestep,
                                kv_cache=group_kvcache,
                                crossattn_cache=group_crossattn_cache,
                                current_start=cur_frame * self.frame_seq_length
                            )

                    # ---- Persist training data (first newly-generated frame only) ----
                    if cur_frame == current_start_frame:
                        group_latents.append(
                            denoised_pred.reshape(
                                batch_size, micro_batch_size, *denoised_pred.shape[1:]
                            )
                        )
                        group_train_latents.append(
                            torch.stack(denoising_latents, dim=1).reshape(
                                batch_size,
                                micro_batch_size,
                                len(denoising_latents),
                                *denoising_latents[0].shape[1:],
                            )
                        )
                        group_log_probs.append(
                            torch.stack(log_probs, dim=1).reshape(
                                batch_size,
                                micro_batch_size,
                                len(log_probs),
                                *log_probs[0].shape[1:],
                            )
                        )
                        group_flow_preds.append(
                            torch.stack(flow_preds, dim=1).reshape(
                                batch_size,
                                micro_batch_size,
                                len(flow_preds),
                                *flow_preds[0].shape[1:],
                            )
                        )

                    output_group[:, cur_frame : cur_frame + current_num_frames] = denoised_pred

                    # ---- Update KV cache with clean (denoised) context ----
                    context_timestep = torch.ones_like(timestep) * self.args.context_noise
                    self.generator(
                        noisy_image_or_video=denoised_pred,
                        conditional_dict=cond_in_use,
                        timestep=context_timestep,
                        kv_cache=group_kvcache,
                        crossattn_cache=group_crossattn_cache,
                        current_start=cur_frame * self.frame_seq_length,
                    )

                    cur_frame += current_num_frames

                # Decode video for this group
                for micro_index in range(micro_batch_size):
                    g_video = self.vae.decode_to_pixel(
                        output_group[
                            micro_index * batch_size : (micro_index + 1) * batch_size
                        ],
                        use_cache=False,
                    )
                    g_video = (g_video * 0.5 + 0.5).clamp(0, 1)
                    videos.append(g_video)
                    self.vae.model.clear_cache()

        del group_crossattn_cache
        del group_kvcache
        del kv_cache1

        # Stacked prompt embeddings: each segment's embedding as one slice along last dim
        prompt_embedings = torch.stack([c["prompt_embeds"] for c in cond_list], dim=-1)

        collate_data = {
            "video": torch.stack(videos, dim=1),              # B, G, F, C, H, W
            "latent": output,
            "prompt_embedings": prompt_embedings,
            "diffusion_log_prob": torch.cat(group_log_probs, dim=1),        # B, G, T, F, C, H, W
            "diffusion_train_latents": torch.cat(group_train_latents, dim=1),  # B, G, T, F, C, H, W
            "diffusion_flow_preds": torch.cat(group_flow_preds, dim=1),       # B, G, T, F, C, H, W
            "rewritting_prompt": rewritting_prompt,
            "llm_generation_ids": llm_generation_ids,
            "llm_prompt_ids": llm_prompt_ids,
            "format_reward": format_reward,
            "respones_json_mask": respones_json_mask,
            "prompt_for_reward": prompt_for_reward,
            "prompt_for_llm_rewards": prompt_for_llm_rewards,
        }

        if return_log_prob:
            collate_data["old_gen_log_probs"] = llm_feature["old_gen_log_probs"]

        return collate_data 