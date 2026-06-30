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

# OpenAI-protocol config. Set these via environment variables.
APP_ID = os.environ.get("GEMINI_APP_ID", "")
APP_KEY = os.environ.get("GEMINI_APP_KEY", "")
BASE_URL = os.environ.get("GEMINI_BASE_URL", "")
MODEL = "api_naci_default_gemini-3-flash-preview"
# MODEL = "api_naci_default_gemini-2.5-pro"

URL = BASE_URL.rstrip("/") + "/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {APP_ID}:{APP_KEY}",
}

def clean_json_output(text):
    # 去掉 ```json ``` 包裹
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()

def is_video_path(video_input):
    """
    判断输入是视频文件路径还是 tensor
    """
    if isinstance(video_input, str) and os.path.isfile(video_input):
        return True
    return False

def extract_gemini_json(output, num_sample_frames):
    """
    从 Gemini 输出中提取 JSON 部分，并返回 frame-wise reward tensor
    """
    try:
        # 匹配 ```json ... ``` 中的内容
        match = re.search(r"```json\s*(\{.*?\})\s*```", output, re.DOTALL)
        print(output)
        if match:
            cleaned = match.group(1)
            data = json.loads(cleaned)
            
            frame_scores = np.array(data.get("frame_scores", [0]*num_sample_frames), dtype=np.float32)/10.0
            global_score = float(data.get("final_score", 0))/10.0

            temporal_order_score = float(data.get("temporal_order_score", 0))/10.0
            physical_plausibility_score = float(data.get("physical_plausibility_score", 0))/10.0
            visual_consistency_score = float(data.get("visual_consistency_score", 0))/10.0
            prompt_alignment_score = float(data.get("prompt_alignment_score", 0))/10.0

            # 综合局部 + 全局
            alpha = 0.2
            beta = 0.3
            beta_1 = 0.2
            beta_2 = 0.1
            beta_3 = 0.1
            beta_4 = 0.1
            combined = alpha * frame_scores + beta * global_score + beta_1 * temporal_order_score + beta_2 * physical_plausibility_score + beta_3 * visual_consistency_score + beta_4 * prompt_alignment_score
        else:
            # 如果没有匹配到 JSON，返回全 0
            combined = np.zeros(num_sample_frames, dtype=np.float32)

    except Exception as e:
        print(f"Warning: failed to parse Gemini output: {e}")
        combined = np.zeros(num_sample_frames, dtype=np.float32)

    return torch.tensor(combined, dtype=torch.float32)

def gemini_video_score(device):

    def extract_frames_from_video(video_path, num_frames=8):
        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(total_frames // num_frames, 1)

        frames = []

        for i in range(num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
            ret, frame = cap.read()

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        cap.release()
        return frames

    def frames_to_base64(frames):
        """将帧列表转换为base64编码的图像列表"""
        base64_images = []
        for frame in frames:
            img = Image.fromarray(frame)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            base64_images.append(f"data:image/jpeg;base64,{base64_str}")
        return base64_images

    def build_prompt(prompt):
        TEXT = f"""
You are a video reasoning expert evaluating whether a generated video follows a textual prompt.

You will be given:
1. A text prompt describing a sequence of actions.
2. A set of video frames sampled in chronological order.

Your task is to analyze the frames and evaluate the video.

Step 1 — Event Decomposition
Break the prompt into atomic events that must occur.

Step 2 — Event Verification
For each atomic event:
- Determine whether the event occurs in the video
- Score the execution quality from 0-10

Step 3 — Temporal Reasoning
Check:
- Whether the events occur in the correct order
- Whether the transitions between events are reasonable

Step 4 — Physical Plausibility
Check whether the actions follow basic physical rules.

Step 5 — Visual Consistency
Check whether objects and actors remain consistent across frames.

Step 6 — Overall Prompt Faithfulness
Evaluate how well the video follows the prompt.

PROMPT:
{prompt}

Output strictly in JSON format:

{{
  "events": [
    {{
      "event": "...",
      "occurred": true/false,
      "score": 0-10,
      "explanation": "..."
    }}
  ],

  "temporal_order_score": 0-10,
  "physical_plausibility_score": 0-10,
  "visual_consistency_score": 0-10,
  "prompt_alignment_score": 0-10,
  "final_score": 0-10
}}
"""
        return TEXT


    def build_prompt_framewise(prompt, num_sample_frames):
        TEXT = f"""
You are a video reasoning expert evaluating whether a generated video follows a textual prompt.

You will be given:
1. A text prompt describing a sequence of actions.
2. A set of {num_sample_frames} video frames sampled in chronological order.

Your task is to evaluate the video strictly according to the following JSON schema:

{{
  "frame_scores": [0, 0, 0, 0, 0, 0, 0, 0],   // length = {num_sample_frames}, 0-10 per frame
  "temporal_order_score": 0,                 // 0-10
  "physical_plausibility_score": 0,          // 0-10
  "visual_consistency_score": 0,             // 0-10
  "prompt_alignment_score": 0,               // 0-10
  "final_score": 0,                          // 0-10
  "think": {{                                // optional, textual explanations
      "temporal_reasoning": "",
      "physical_plausibility": "",
      "visual_consistency": "",
      "prompt_faithfulness": ""
  }}
}}

Requirements:
- Output must be **strictly valid JSON** matching the schema above.
- Scores must be integers from 0 to 10.
- "frame_scores" must have exactly {num_sample_frames} elements, in chronological order.
- "think" field should contain optional text reasoning, can include explanations per frame and overall reasoning.
- Do NOT include extra fields outside the schema.

PROMPT:
{prompt}
"""
        
        WITHOUT_THINK_TEXT = f"""
You are a video reasoning expert evaluating whether a generated video follows a textual prompt.

You will be given:
1. A text prompt describing a sequence of actions.
2. A set of {num_sample_frames} video frames sampled in chronological order.

Your task is to evaluate the video strictly according to the following JSON schema:

{{
  "frame_scores": [0, 0, 0, 0, 0, 0, 0, 0],   // length = {num_sample_frames}, 0-10 per frame
  "temporal_order_score": 0,                 // 0-10
  "physical_plausibility_score": 0,          // 0-10
  "visual_consistency_score": 0,             // 0-10
  "prompt_alignment_score": 0,               // 0-10
  "final_score": 0,                          // 0-10
}}

Requirements:
- Output must be **strictly valid JSON** matching the schema above.
- Scores must be integers from 0 to 10.
- "frame_scores" must have exactly {num_sample_frames} elements, in chronological order.
- "think" field should contain optional text reasoning, can include explanations per frame and overall reasoning.
- Do NOT include extra fields outside the schema.
- Do NOT include explanations.

PROMPT:
{prompt}
"""
        SIMPLE_TEXT = f"""
Fill in the JSON with scores for how well the video matches the prompt.

Return JSON:

{{
  "frame_scores": [],
  "temporal_order_score": 0,
  "physical_plausibility_score": 0,
  "visual_consistency_score": 0,
  "prompt_alignment_score": 0,
  "final_score": 0
}}

Rules:
- All scores are integers 0-10
- frame_scores length = {num_sample_frames}
- Output JSON only
- No explanation
- No reasoning
- Do not think step by step
- Output directly

PROMPT:
{prompt}
"""
        return TEXT

    def call_openai_api(frames, prompt, max_retries=3):
        """使用OpenAI协议调用Gemini模型"""
        try_num = 0
        output = ""

        while output == "" and try_num <= max_retries:
            try_num += 1

            try:
                # 将帧转换为base64
                base64_images = frames_to_base64(frames)
                
                # 构建OpenAI格式的消息
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt}
                        ] + [
                            {"type": "image_url", "image_url": {"url": img_url}} 
                            for img_url in base64_images
                        ]
                    }
                ]

                # 构建请求体
                body = {
                    "model": MODEL,
                    "messages": messages,
                    "stream": False,
                    "temperature": 0.0,
                    "top_p": 1.0
                }

                # 发送请求
                response = requests.post(URL, headers=HEADERS, json=body, timeout=80)
                
                if response.status_code == 200:
                    data = response.json()
                    if "usage" in data:
                        usage = data["usage"]
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        total_tokens = usage.get("total_tokens", 0)
                        print(f"[Token Usage] prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")


                    if data and data.get("choices"):
                        msg = data["choices"][0].get("message")
                        if msg and msg.get("content"):
                            output = msg["content"]
                            # 如果有思维链内容，也包含在内
                            if msg.get("reasoning_content"):
                                output += "\n" + msg["reasoning_content"]
                        else:
                            output = str(data)
                    else:
                        output = str(data)
                else:
                    print(f"API调用失败，状态码: {response.status_code}, 响应: {response.text}")
                    output = ""

                return output

            except Exception as e:
                print(f"OpenAI API error try {try_num}: {e}")
                if try_num < max_retries:
                    time.sleep(2)

        return ""


    def _fn(video_inputs, prompts, metadata=None):
        """
        video_tensors: torch.Tensor, [B, F, C, H, W]
        prompts: list[str], length B
        return: scores: np.ndarray [B, num_sample_frames], range [0,1]
        """
        # num_sample_frames=8
        num_sample_frames = metadata[0].get("eval_sample_frames", 6)
        K = 1

        B = len(video_inputs)
        scores_matrix = torch.zeros((B, num_sample_frames), dtype=torch.float32)
        if isinstance(video_inputs, torch.Tensor):
            video_inputs = (video_inputs * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()

        outputs = []
        for i, video in enumerate(video_inputs):
            if is_video_path(video):
                # 视频文件路径 → 读取帧
                frames = extract_frames_from_video(video, num_frames=num_sample_frames)
            else:
                # 视频 tensor → 抽帧
                # 假设 shape [F,C,H,W]

                F, C, H, W = video.shape
                frames = []
                start_idx = 0
                # interval = 12
                interval = max(F // num_sample_frames, 1)
                for f_idx in range(num_sample_frames):
                    idx = min(start_idx + f_idx * interval, F-1)
                    frame = video[idx].transpose(1,2,0)  # C,H,W -> H,W,C
                    frames.append(frame)


            # 调用 Gemini
            # import ipdb; ipdb.set_trace()
            prompt_text = build_prompt_framewise(prompts[i], num_sample_frames)

            all_video_rewards = []

            for k in range(K):  # K=3 or 5
                output = call_openai_api(frames, prompt_text)
                frame_tensor = extract_gemini_json(output, num_sample_frames)

                # video_reward = frame_tensor.mean()
                all_video_rewards.append(frame_tensor[:num_sample_frames])
            scores_matrix[i] = torch.stack(all_video_rewards, dim=0).mean(0)

            outputs.append(output)

        scores_matrix = scores_matrix.mean(-1)
        return scores_matrix, {"outputs": outputs}

    return _fn
