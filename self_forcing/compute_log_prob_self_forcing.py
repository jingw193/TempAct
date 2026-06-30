import torch
import math
import torch.nn.functional as F
import json

def compute_log_prob(
        noisy_input, # x_t
        next_noisy_input, # x_t_1
        flow_pred, 
        current_timestep, 
        next_timestep,
    ):

    log_prob = (
            -((next_noisy_input.detach() - (1 - next_timestep) * (noisy_input - flow_pred*current_timestep)) ** 2) / (2 * ((next_timestep)**2))
            - torch.log(next_timestep)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
    # log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    return log_prob


def llm_rewritting_prompt(llm_policy, llm_policy_tokenizer, usr_prompts, gap_frame, sample_frame, return_log_prob=False):
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

    # diverse prompt for grpo training
    detail_instruction_prompt = """
You are an expert in temporal reasoning for video generation.

Think Content Rules:
1. Limit think content to at most 256 tokens.
2. Avoid repeating the same reasoning or descriptions.
3. Use concise and varied phrasing.
4. Focus only on reasoning necessary to generate step prompts.
5. You may explore alternative wordings or slightly enrich descriptions to improve temporal clarity.

A video diffusion model generates a video progressively over time. 
Given the original prompt describing the entire video, split the video evenly into {total_steps} sequential steps 
and rewrite the prompt for each step so that it describes the scene and actions that should logically appear during that step.

Step Prompt Rules:
1. Maintain logical consistency with the original prompt.
2. Use present tense.
3. Do NOT describe past or future events outside this step.
4. Do NOT add new characters, objects, or events.
5. You MAY slightly vary phrasing, enrich details, or add natural descriptive variation within the step.
6. Keep prompts concise but informative to guide video generation effectively.

Output format:
Return a JSON array with {total_steps} elements, each element containing:
- "step_index": step index (0-based)
- "prompt": a detailed rewritten prompt for this step, capturing actions, visual details, and transitions

Input:
Original prompt:
{original_prompt}
"""


    global_instruction_prompt = """
You are an expert in rewriting prompts for video generation models.

Goal:
Rewrite the original prompt to improve its suitability for video diffusion models,
with a focus on temporal coherence, motion consistency, and physical plausibility.

Think Content Rules:
1. Limit reasoning to at most 256 tokens.
2. Avoid repetition.
3. Focus only on necessary transformations.

Optimization Objectives:

1. Temporal Coherence:
- Ensure actions are described in a clear and logical order.
- Make implicit temporal relationships explicit when necessary.

2. Motion Consistency:
- Maintain a consistent subject and continuous motion.
- Avoid abrupt or unrealistic transitions.

3. Physical Plausibility:
- Ensure actions and interactions follow basic physical intuition.
- Avoid contradictory or impossible descriptions.

Diversity Requirements:
1. Vary sentence structure and phrasing.
2. You may adjust emphasis (motion, appearance, environment).
3. You may slightly enrich visual details (e.g., lighting, texture),
   but MUST NOT introduce new objects, characters, or events.
4. Avoid rigid or templated phrasing.

Global Prompt Rules:
1. Preserve the original semantics.
2. Use present tense.
3. Do NOT add new entities or actions.
4. Improve clarity for video generation.
5. Keep the prompt concise but expressive.

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
        # instruction_prompt = detail_instruction_prompt
        instruction_prompt = PE_detail_instruction_prompt
    else:
        total_steps = 1
        instruction_prompt = global_instruction_prompt


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