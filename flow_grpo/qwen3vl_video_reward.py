"""
Qwen3-VL-8B-Thinking-FP8 Video Reward Server
基于本地 vllm serve 的 OpenAI 兼容接口

vllm 启动示例:
  vllm serve Qwen/Qwen3-VL-8B-Thinking-FP8 \
      --host 127.0.0.1 --port 8000 \
      --tensor-parallel-size 1 \
      --max-model-len 32768 \
      --limit-mm-per-prompt image=16 \
      --trust-remote-code

调用方式同 gemini_reward.py: 输入 (video_inputs, prompts, metadata)，返回 (scores_tensor, info_dict)
"""

import os
import io
import cv2
import json
import time
import torch
import numpy as np
from PIL import Image
import re
import requests
import base64
from copy import deepcopy
# ──────────────────────────────────────────────────────────────
# vllm serve 配置：与 gemini_reward.py 中的 OpenAI 协议保持一致
# ──────────────────────────────────────────────────────────────
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
print(f"VLLM_BASE_URL: {VLLM_BASE_URL}")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")  # vllm 默认不校验 key

URL = VLLM_BASE_URL.rstrip("/") + "/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {VLLM_API_KEY}",
}


# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────

def is_video_path(video_input):
    """判断输入是视频文件路径还是 numpy array / tensor"""
    return isinstance(video_input, str) and os.path.isfile(video_input)


def extract_frames_from_video(
    video_path: str,
    num_frames: int = 8,
    start_frame: int = 0,
    end_frame: int = None
):
    """从视频指定帧范围内均匀抽取 num_frames 帧，返回 np.ndarray list (H,W,3 RGB)"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return []

    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames

    start_frame = max(0, start_frame)
    end_frame = max(start_frame + 1, end_frame)

    valid_length = end_frame - start_frame
    interval = max(valid_length // num_frames, 1)

    frames = []
    for i in range(num_frames):
        frame_idx = start_frame + i * interval
        if frame_idx >= end_frame:
            frame_idx = end_frame - 1

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()

    # 如果实际读取帧数不足，用最后一帧填充
    while len(frames) < num_frames and len(frames) > 0:
        frames.append(frames[-1])

    return frames

def frames_to_base64(frames):
    """将帧列表（np.ndarray, RGB uint8）转换为 base64 data-url 列表"""
    base64_images = []
    for frame in frames:
        img = Image.fromarray(frame.astype(np.uint8))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        base64_images.append(f"data:image/jpeg;base64,{base64_str}")
    return base64_images


# ──────────────────────────────────────────────────────────────
# JSON 解析：兼容 Qwen3 thinking 格式（<think>...</think>）
# ──────────────────────────────────────────────────────────────

def strip_thinking_tags(text: str) -> str:
    """去除 Qwen3 thinking 标签 <think>...</think>，保留最终答案部分"""
    # 去除 <think> 块
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()

def extract_json_from_output_clipwise(output: str):
    """
    从模型输出中提取 JSON, 解析 clip-level 分数并返回一个标量 tensor。

    支持三种格式：
      1. ```json { ... } ``` 代码块
      2. 裸 JSON: { ... }
      3. 解析失败时返回 0 分
    """
    try:
        clean = strip_thinking_tags(output)

        match = re.search(r"```json\s*(\{.*?\})\s*```", clean, re.DOTALL)
        if not match:
            match = re.search(r"(\{.*\})", clean, re.DOTALL)

        if match:
            data = json.loads(match.group(1))

            subject_score = float(data.get("subject_score", 0)) / 10.0
            scene_score = float(data.get("scene_score", 0)) / 10.0
            action_score = float(data.get("action_score", 0)) / 10.0
            object_appearance_reasonability_score = float(data.get("object_appearance_reasonability_score", 0)) / 10.0
            final_score = float(data.get("final_score", 0)) / 10.0

            combined = (
                0.15 * subject_score
                + 0.15 * scene_score
                + 0.2 * action_score
                + 0.2 * object_appearance_reasonability_score
                + 0.3 * final_score
            )
        else:
            print("Warning: no JSON found in model output")
            combined = 0.0

    except Exception as e:
        print(f"Warning: failed to parse model output: {e}")
        combined = 0.0

    return torch.tensor(combined, dtype=torch.float32)

def extract_json_from_output(output: str, num_sample_frames: int):
    """
    从模型输出中提取 JSON,解析为评分数据并返回 combined tensor。

    支持三种格式：
      1. ```json { ... } ``` 代码块
      2. 裸 JSON:{ ... }
      3. 解析失败时返回全零 tensor
    """
    try:
        # 先去掉 thinking 内容
        clean = strip_thinking_tags(output)

        # 尝试匹配 ```json ... ``` 代码块
        match = re.search(r"```json\s*(\{.*?\})\s*```", clean, re.DOTALL)
        if not match:
            # 尝试裸 JSON
            match = re.search(r"(\{.*\})", clean, re.DOTALL)

        # print("[Qwen3-VL output]", clean[:500])

        if match:
            data = json.loads(match.group(1))

            frame_scores = np.array(
                data.get("frame_scores", [0] * num_sample_frames), dtype=np.float32
            ) / 10.0
            # 对齐帧数
            if len(frame_scores) < num_sample_frames:
                frame_scores = np.pad(
                    frame_scores, (0, num_sample_frames - len(frame_scores)), constant_values=0.0
                )
            else:
                frame_scores = frame_scores[:num_sample_frames]

            global_score              = float(data.get("final_score", 0)) / 10.0
            temporal_order_score      = float(data.get("temporal_order_score", 0)) / 10.0
            physical_plausibility_score = float(data.get("physical_plausibility_score", 0)) / 10.0
            visual_consistency_score  = float(data.get("visual_consistency_score", 0)) / 10.0
            prompt_alignment_score    = float(data.get("prompt_alignment_score", 0)) / 10.0

            # 与 gemini_reward.py 保持相同的加权方式
            alpha  = 0.2   # frame-level weight
            beta   = 0.3   # global final_score
            beta_1 = 0.2   # temporal order
            beta_2 = 0.1   # physical plausibility
            beta_3 = 0.1   # visual consistency
            beta_4 = 0.1   # prompt alignment

            combined = (
                alpha  * frame_scores
                + beta   * global_score
                + beta_1 * temporal_order_score
                + beta_2 * physical_plausibility_score
                + beta_3 * visual_consistency_score
                + beta_4 * prompt_alignment_score
            )
        else:
            print("Warning: no JSON found in Qwen3-VL output")
            combined = np.zeros(num_sample_frames, dtype=np.float32)

    except Exception as e:
        print(f"Warning: failed to parse Qwen3-VL output: {e}")
        combined = np.zeros(num_sample_frames, dtype=np.float32)

    return torch.tensor(combined, dtype=torch.float32)


def build_prompt_clipwise_no_think_smoothprompt(prompt: str, num_sample_frames: int) -> str:
    return f"""You are a strict video-text matching evaluator.

Determine whether a short video segment matches the given text description using ONLY visible evidence from {num_sample_frames} sampled frames in chronological order.

The text description is a temporally localized video caption describing what should visually happen during this video segment.

Evaluation Rules:
1. Use only visible evidence from the sampled frames.
2. Do NOT guess unseen details or hidden motion.
3. Focus on whether the described subjects, scene, objects, motion, and visual dynamics are clearly visible.
4. The segment does NOT need to contain a complete action lifecycle if the visible partial progression matches the description naturally.
5. Judge both appearance accuracy and temporal plausibility.
6. If details are unclear, ambiguous, partially visible, or weakly supported, assign lower scores.
7. If there is a clear contradiction between the text and frames, assign very low scores.
8. Superficial similarity is not enough for a high score.
9. High scores require strong positive visual evidence.
10. When uncertain between two score ranges, choose the lower range.
11. Do not average away major errors or contradictions.
12. If a key required detail is unclear or incomplete, the relevant score should usually not exceed 6.
13. If a key subject, object, scene, motion, or interaction is missing, the relevant score should not exceed 4.
14. If the described motion or action is not directly visible, action_score should not exceed 4.
15. If motion progression appears temporally abrupt, physically implausible, or visually inconsistent, object_appearance_reasonability_score should be reduced.
16. final_score >= 8 requires clear alignment across all major dimensions with no major contradiction.
17. Strong scores require both semantic correctness and visually coherent temporal evolution.

Also evaluate the visual reasonability of how text-required object(s), subject(s), and motion appear across frames.
A high object_appearance_reasonability_score means:
- objects appear naturally and consistently
- motion evolves smoothly across frames
- temporal progression feels visually coherent
- object interactions follow plausible physical behavior

A low score means:
- objects appear abruptly or disappear unnaturally
- motion lacks temporal continuity
- frame-to-frame evolution is implausible or inconsistent
- interactions appear visually broken or unsupported

Score from 0 to 10:
- subject_score
- scene_score
- action_score
- object_appearance_reasonability_score
- final_score

Strict score interpretation:
- 9-10: almost exact match; all key required elements are clearly visible, temporally coherent, and visually consistent
- 7-8: strong match with only minor non-critical imperfections
- 5-6: partial match; at least one important requirement is unclear, incomplete, weakly supported, or temporally inconsistent
- 3-4: poor match; only superficial similarity or limited overlap
- 1-2: very poor match; most key requirements are absent or unsupported
- 0: clear contradiction

Return ONLY valid JSON:
{{
  "think": {{
      "detected_errors": [
          "..."
      ],
      "temporal_reasoning": [
          "..."
      ],
      "motion_smooth_and_blur_reasoning": [
          "..."
      ]
  }},
  "subject_score": 0,
  "scene_score": 0,
  "action_score": 0,
  "object_appearance_reasonability_score": 0,
  "final_score": 0
}}

Text description:
{prompt}
"""

def build_prompt_framewise(prompt: str, num_sample_frames: int) -> str:
    return f"""You are a video consistency evaluator.

Your task is to determine how well a video matches a text prompt using STRICT evidence-based reasoning.

---

INPUT:
- A text prompt describing actions
- {num_sample_frames} frames in chronological order

---

CRITICAL PRINCIPLE:

You MUST base ALL scoring on explicit observable evidence from frames.

Do NOT:
- give scores based on overall impression
- guess missing errors without visual evidence
- assign uniformly high or low scores without justification

---

STEP 1: FRAME-BY-FRAME OBSERVATION

For EACH frame:
- describe visible objects and actions
- check consistency with previous frame
- check consistency with prompt

You MUST explicitly note:
- any object appearing/disappearing
- any motion inconsistency
- any identity change

---

STEP 2: LOCAL ERROR DETECTION

For EACH frame, mark:

- "OK"
- "Minor issue"
- "Major issue"
- "Critical failure"

BUT only if there is visible evidence.

---

STEP 3: FRAME SCORE ASSIGNMENT RULE

Each frame_score[i] is determined ONLY by:

- 9-10: fully consistent frame, no visible issues
- 7-8: minor ambiguity or slight motion inconsistency
- 4-6: clear but not severe issue
- 1-3: strong inconsistency (wrong object, broken motion)
- 0: frame is visually incoherent or contradicts previous frame

IMPORTANT:
- Scores MUST reflect local evidence
- Do NOT reuse same score unless frames are truly identical

---

STEP 4: GLOBAL SCORE COMPUTATION

Global scores MUST be computed ONLY from:
- number of "Major/Critical" issues
- frequency of inconsistencies
- coverage of prompt actions

No direct subjective judgment allowed.

---

SCORING ANCHOR (VERY IMPORTANT):

- Perfect video: 9–10
- Mostly correct with minor noise: 7–8
- Noticeable errors (missing / wrong actions): 5–6
- Severe temporal/physical errors: 2–4
- Almost meaningless video: 0–1

---

OUTPUT FORMAT (STRICT JSON):

```json
{{
  "think": {{
      "frame_evidence": [
          "frame 1: ...",
          "frame 2: ...",
          "frame 3: ..."
      ],
      "detected_errors": [
          "..."
      ],
      "aggregation_logic": ""
  }},
  "frame_scores": [0, 0, 0, 0, 0, 0],
  "temporal_order_score": 0,
  "physical_plausibility_score": 0,
  "visual_consistency_score": 0,
  "prompt_alignment_score": 0,
  "final_score": 0
}}
```

---
PROMPT:
{prompt}
"""

# ──────────────────────────────────────────────────────────────
# vllm API 调用（OpenAI 兼容，与 gemini_reward.py call_openai_api 一致）
# ──────────────────────────────────────────────────────────────

def call_vllm_api(frames, prompt_text: str, max_retries: int = 3) -> str:
    """
    通过 OpenAI 兼容接口调用本地 vllm serve 的 Qwen3-VL 模型。

    图像以 base64 data-url 方式传入，与 gemini_reward.py 相同。
    Qwen3-VL 特有的 thinking 内容会附加在 reasoning_content 字段中。
    """
    try_num = 0
    output = ""

    while output == "" and try_num <= max_retries:
        try_num += 1
        try:
            base64_images = frames_to_base64(frames)

            # OpenAI 多模态消息格式
            content = [{"type": "text", "text": prompt_text}]
            for img_url in base64_images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img_url}
                })

            messages = [{"role": "user", "content": content}]

            body = {
                "model": VLLM_MODEL,
                "messages": messages,
                "stream": False,
                "temperature": 0.0,
                "top_p": 1.0,
                # Qwen3 thinking 模式：开启推理（vllm 支持）
                # 如果 vllm 不支持此字段会被忽略，不影响正常调用
                "chat_template_kwargs": {"enable_thinking": True},
            }

            response = requests.post(URL, headers=HEADERS, json=body, timeout=120)

            if response.status_code == 200:
                data = response.json()
                # 打印 token 用量
                if "usage" in data:
                    usage = data["usage"]

                if data and data.get("choices"):
                    msg = data["choices"][0].get("message", {})
                    content_text = msg.get("content") or ""
                    # Qwen3 thinking 内容（若有）拼接到输出，供 extract_json_from_output 处理
                    reasoning = msg.get("reasoning_content") or ""
                    output = content_text
                    if reasoning:
                        # 将 thinking 内容包裹成 <think> 标签格式，与 strip_thinking_tags 兼容
                        output = f"<think>{reasoning}</think>\n{content_text}"
                else:
                    output = str(data)

                return output
            else:
                print(
                    f"vllm API 调用失败, status={response.status_code}, "
                    f"resp={response.text[:200]}"
                )
                output = ""

        except Exception as e:
            print(f"vllm API error try {try_num}: {e}")
            if try_num <= max_retries:
                time.sleep(2)

    return ""


# ──────────────────────────────────────────────────────────────
# 主推理函数工厂（与 gemini_reward.py gemini_video_score 接口完全一致）
# ──────────────────────────────────────────────────────────────

def qwen3vl_video_score(device: str = "cuda"):
    """
    返回与 gemini_video_score 完全相同签名的推理函数 _fn。

    _fn(video_inputs, prompts, metadata) -> (scores_tensor [B], info_dict)

    video_inputs: list of
        - str: 视频文件路径
        - np.ndarray: [F, C, H, W] uint8 或 float32 (值域 0~1)
        - torch.Tensor: [F, C, H, W] float32 (值域 0~1)
    prompts: list[str], length B
    metadata: list[dict], 支持字段:
        - eval_sample_frames (int, default=6): 抽取帧数
        - num_votes (int, default=1): 重复调用次数取平均（集成投票）
    """

    def _fn(video_inputs, prompts, metadata=None):
        if metadata is None:
            metadata = [{}] * len(prompts)

        num_sample_frames = metadata[0].get("eval_sample_frames", 6)
        K = metadata[0].get("num_votes", 1)  # 投票次数，默认 1
        start_frame = metadata[0].get("start_frame", 0)  # 开始帧，默认 0

        B = len(video_inputs)
        scores_matrix = torch.zeros((B, num_sample_frames), dtype=torch.float32)

        # 如果是 torch.Tensor，先转 numpy
        if isinstance(video_inputs, torch.Tensor):
            video_inputs = (
                video_inputs * 255
            ).round().clamp(0, 255).to(torch.uint8).cpu().numpy()

        outputs = []
        for i, video in enumerate(video_inputs):
            if is_video_path(video):
                # 路径模式：从视频文件读帧
                frames = extract_frames_from_video(video, num_frames=num_sample_frames, start_frame=start_frame)
            else:
                # Tensor/Array 模式：video shape [F, C, H, W]
                if isinstance(video, torch.Tensor):
                    video = (video * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
                elif video.dtype != np.uint8:
                    video = (video * 255).round().clip(0, 255).astype(np.uint8)

                F, C, H, W = video.shape
                interval = max(F // num_sample_frames, 1)
                frames = []
                for f_idx in range(num_sample_frames):
                    idx = min(f_idx * interval, F - 1)
                    frame = video[idx].transpose(1, 2, 0)  # C,H,W -> H,W,C
                    frames.append(frame)

            # 构建 prompt
            prompt_text = build_prompt_framewise(prompts[i], num_sample_frames)

            # K 次投票取平均
            all_video_rewards = []
            last_output = ""
            for k in range(K):
                raw_output = call_vllm_api(frames, prompt_text)
                frame_tensor = extract_json_from_output(raw_output, num_sample_frames)
                all_video_rewards.append(frame_tensor[:num_sample_frames])
                last_output = raw_output

            scores_matrix[i] = torch.stack(all_video_rewards, dim=0).mean(0)
            outputs.append(last_output)

        # 对帧维度取均值，最终返回 [B] 的分数向量
        scores = scores_matrix.mean(-1)
        if "train_frames_ratio" in metadata[0].keys():
            local_scores = scores_matrix[:, int(metadata[0]["train_frames_ratio"] * num_sample_frames):].mean(-1)
            return torch.stack([scores, local_scores], dim=-1), {"outputs": outputs}
        else:
            return scores, {"outputs": outputs}

    return _fn

def qwen3vl_video_local_score(device: str = "cuda"):
    """
    返回与 gemini_video_score 完全相同签名的推理函数 _fn。

    _fn(video_inputs, prompts, metadata) -> (scores_tensor [B], info_dict)

    video_inputs: list of
        - str: 视频文件路径
        - np.ndarray: [F, C, H, W] uint8 或 float32 (值域 0~1)
        - torch.Tensor: [F, C, H, W] float32 (值域 0~1)
    prompts: list[str], length B
    metadata: list[dict], 支持字段:
        - eval_sample_frames (int, default=6): 抽取帧数
        - num_votes (int, default=1): 重复调用次数取平均（集成投票）
    """

    def _fn(video_inputs, prompts, metadata=None):
        if metadata is None:
            metadata = [{}] * len(prompts)

        # decomposed_outputs = [x["decomposed_outputs"] for x in metadata]
        decomposed_outputs = [x["decomposed_llm_outputs"] for x in metadata]
        num_sample_frames = metadata[0].get("eval_sample_frames", 6)
        K = metadata[0].get("num_votes", 1)  # 投票次数，默认 1
        train_frames = metadata[0].get("train_frames", 15)
        eval_frames_region = metadata[0].get("gap_frame", 6) * 4
        prompt_step_idx = train_frames // metadata[0].get("gap_frame", 6)

        start_frame = (train_frames - 1) * 4 + 1

        B = len(video_inputs)
        scores_matrix = torch.zeros((B), dtype=torch.float32)
        # import ipdb; ipdb.set_trace()
        # 如果是 torch.Tensor，先转 numpy
        if isinstance(video_inputs, torch.Tensor):
            video_inputs = video_inputs[:, start_frame:start_frame + eval_frames_region]
            video_inputs = (
                video_inputs * 255
            ).round().clamp(0, 255).to(torch.uint8).cpu().numpy()

        outputs = []
        for i, video in enumerate(video_inputs):
            if is_video_path(video):
                # 路径模式：从视频文件读帧
                frames = extract_frames_from_video(
                    video, 
                    num_frames=num_sample_frames, 
                    start_frame=start_frame,
                    end_frame=start_frame + eval_frames_region
                )
            else:
                # Tensor/Array 模式：video shape [F, C, H, W]
                if isinstance(video, torch.Tensor):
                    video = (video * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
                elif video.dtype != np.uint8:
                    video = (video * 255).round().clip(0, 255).astype(np.uint8)

                F, C, H, W = video.shape
                interval = max(F // num_sample_frames, 1)
                frames = []
                for f_idx in range(num_sample_frames):
                    idx = min(f_idx * interval, F - 1)
                    frame = video[idx].transpose(1, 2, 0)  # C,H,W -> H,W,C
                    frames.append(frame)

            # 构建 prompt
            
            prompt_text = build_prompt_clipwise_no_think_smoothprompt(decomposed_outputs[i][prompt_step_idx]["prompt"], num_sample_frames=num_sample_frames)

            # K 次投票取平均
            all_video_rewards = []
            last_output = ""
            for k in range(K):
                raw_output = call_vllm_api(frames, prompt_text)
                frame_tensor = extract_json_from_output_clipwise(raw_output)
                all_video_rewards.append(frame_tensor)
                last_output = raw_output

            scores_matrix[i] = torch.stack(all_video_rewards, dim=0).mean(0)
            outputs.append(last_output)

        # 对帧维度取均值，最终返回 [B] 的分数向量
        scores = scores_matrix
        return scores, {"outputs": outputs}

    return _fn

