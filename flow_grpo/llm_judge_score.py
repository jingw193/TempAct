import json
from typing import List, Dict, Any, Callable, Optional
import time
import torch
import requests
import os
import re

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000")
print(f"LLM_BASE_URL: {LLM_BASE_URL}")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen3-8B")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "EMPTY")  # vllm 默认不校验 key

LLM_URL = LLM_BASE_URL.rstrip("/") + "/v1/chat/completions"
LLM_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {LLM_API_KEY}",
}

def call_llm_api(prompt_text: str, max_retries: int = 3) -> str:
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

            content = [{"type": "text", "text": prompt_text}]

            messages = [{"role": "user", "content": content}]

            body = {
                "model": LLM_MODEL,
                "messages": messages,
                "stream": False,
                "temperature": 0.0,
                "top_p": 1.0,
                # Qwen3 thinking 模式：开启推理（vllm 支持）
                # 如果 vllm 不支持此字段会被忽略，不影响正常调用
                "chat_template_kwargs": {"enable_thinking": True},
            }

            response = requests.post(LLM_URL, headers=LLM_HEADERS, json=body, timeout=120)

            if response.status_code == 200:
                data = response.json()

                # 打印 token 用量
                if "usage" in data:
                    usage = data["usage"]
                    # print(
                    #     f"[Token Usage] prompt={usage.get('prompt_tokens', 0)}, "
                    #     f"completion={usage.get('completion_tokens', 0)}, "
                    #     f"total={usage.get('total_tokens', 0)}"
                    # )

                if data and data.get("choices"):
                    msg = data["choices"][0].get("message", {})
                    content_text = msg.get("content") or ""
                    # Qwen3 thinking 内容（若有）拼接到输出，供 extract_json_from_output 处理
                    reasoning = msg.get("reasoning_content") or ""
                    output = content_text
                    # if reasoning:
                    #     # 将 thinking 内容包裹成 <think> 标签格式，与 strip_thinking_tags 兼容
                    #     output = f"<think>{reasoning}</think>\n{content_text}"
                else:
                    output = str(data)

                return output
            else:
                print(
                    f"llm API 调用失败, status={response.status_code}, "
                    f"resp={response.text[:200]}"
                )
                output = ""

        except Exception as e:
            print(f"llm API error try {try_num}: {e}")
            if try_num <= max_retries:
                time.sleep(2)

    return ""

def decomposition_reward_score(device):
    """
    返回一个批量 reward 评分函数 _fn

    _fn(original_prompts, decomposed_outputs, metadata=None) -> (scores_tensor [B], info_dict)

    original_prompts: list[str]
    decomposed_outputs: list[list[dict]]
        每个元素对应一个样例的 step 列表，格式如：
        [
          {"step_index": 0, "prompt": "..."},
          {"step_index": 1, "prompt": "..."}
        ]

    metadata: list[dict], 支持字段：
        - num_votes (int, default=1): 重复调用 judge 次数取平均
        - return_subscores (bool, default=True): 是否返回子分数
    """

    def _format_decomposition_steps(decomposed_steps: List[Dict[str, Any]]) -> str:
        lines = []
        for step in decomposed_steps:
            idx = step.get("step_index", -1)
            prompt = step.get("prompt", "")
            lines.append(f"[Step {idx}]\n{prompt}\n")
        return "\n".join(lines)

    def _build_judge_single_promptv3(original_prompt: str, decomposed_steps: List[Dict[str, Any]]) -> str:
        formatted_steps = _format_decomposition_steps(decomposed_steps)

        return f"""
You are a careful judge for temporal video prompt decomposition quality.

You are given:
1. an original video prompt
2. an ordered decomposition into step prompts

Each step prompt describes only the current temporal segment of the video.

The decomposition is intended for autoregressive video generation.

Your goal is to judge whether the decomposition is:
- faithful to the original prompt
- temporally coherent
- properly decomposed into meaningful motion stages
- minimally hallucinated
- useful for progressive video generation

Core principle:
A strong decomposition should transform the original prompt into temporally localized video captions that clearly represent evolving motion and scene progression across steps.

Each step should:
- represent a distinct moment or motion stage
- describe only what is visually happening during that temporal segment
- remain visually coherent and temporally localized
- contribute meaningful progression to the sequence

Do NOT reward decompositions that:
- merely split sentences mechanically
- repeat nearly identical prompts across steps
- produce weak or generic progression
- collapse most actions into only one or two steps
- use short symbolic event labels instead of visually meaningful descriptions

Judging policy:
1. Prioritize preservation of the original meaning, event structure, and temporal order.
2. Allow mild paraphrasing if the semantic meaning remains consistent.
3. Allow moderate visual enrichment if it improves motion clarity or video generation quality without changing core semantics.
4. Penalize unsupported additions, especially:
   - new objects or subjects
   - invented actions or events
   - changed causal structure
   - incompatible scene changes
5. Mild cinematic or atmospheric enhancement is acceptable if it remains visually plausible and semantically aligned.
6. Penalize excessive hallucination such as:
   - unsupported colors, materials, weather, or lighting
   - detailed camera language not implied by the prompt
   - exaggerated cinematic additions
   - new scene elements
7. A strong decomposition should distribute event progression across steps instead of concentrating most actions into a single step.
8. Penalize steps that are:
   - overly generic
   - repetitive
   - weakly distinguishable
   - temporally redundant
   - missing meaningful progression
9. Intermediate steps should capture partial motion evolution rather than only discrete start/end events.
10. Adjacent steps should transition naturally and reflect smooth temporal progression.
11. Later steps should continue advancing the sequence rather than becoming repetitive or semantically empty.
12. Do not over-penalize small visual enrichment that improves video generation usability while preserving the original intent.

Common bad patterns:
- Multiple nearly identical steps with only tiny wording differences
- Temporal progression that feels static or mechanically incremental
- Steps that read like short event labels rather than visual video captions
- Abrupt jumps between adjacent steps
- Large unsupported visual or cinematic additions
- Missing intermediate motion evolution
- Repetitive prompts that could be swapped without changing the sequence

Scoring:

1. faithfulness_score
Whether the decomposition preserves the semantic meaning and event structure of the original prompt.

2. coverage_score
Whether all important actions, motion stages, and scene developments are represented across the decomposition steps.

3. temporal_score
Whether the decomposition forms a smooth, logically ordered, temporally coherent progression suitable for autoregressive video generation.

4. hallucination_score
Whether unsupported details or invented content are added.

Hallucination scale:
- 0.0 to 0.2: minor harmless visual enrichment
- 0.2 to 0.5: noticeable unsupported descriptive additions
- 0.5 to 1.0: significant invented details, events, or scene changes

Return ONLY valid JSON:
{{
  "faithfulness_score": float,
  "coverage_score": float,
  "temporal_score": float,
  "hallucination_score": float
}}

Original prompt:
{original_prompt}

Decomposition:
{formatted_steps}
""".strip()


    def _safe_json_load(s: str) -> Dict[str, Any]:
        s = s.strip()

        # 1) 去掉 <think>...</think>
        s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()

        # 2) 去掉 markdown code block
        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?", "", s).strip()
            s = re.sub(r"```$", "", s).strip()

        # 3) 提取第一个完整 JSON 对象
        start = s.find("{")
        if start == -1:
            raise ValueError(f"No JSON object found in output: {s}")

        brace_count = 0
        end = -1
        for i in range(start, len(s)):
            if s[i] == "{":
                brace_count += 1
            elif s[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break

        if end == -1:
            raise ValueError(f"Incomplete JSON object in output: {s}")

        json_str = s[start:end]
        return json.loads(json_str)

    def _clip(x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    def _is_step_index_ordered(decomposed_steps: List[Dict[str, Any]]) -> bool:
        for i, step in enumerate(decomposed_steps):
            if step.get("step_index") != i:
                return False
        return True

    def _fallback_score(original_prompt: str, decomposed_steps: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        没有 judge LLM 时的简单兜底逻辑
        """
        original_words = set(original_prompt.lower().split())
        step_text = " ".join([x.get("prompt", "") for x in decomposed_steps]).lower()
        step_words = set(step_text.split())

        overlap = len(original_words & step_words)
        faithfulness = overlap / max(1, len(original_words))
        coverage = min(1.0, len(decomposed_steps) / 4.0)
        temporal = 1.0 if _is_step_index_ordered(decomposed_steps) else 0.5
        hallucination = max(0.0, len(step_words - original_words) / max(1, len(step_words))) * 0.2

        return {
            "faithfulness_score": _clip(faithfulness),
            "coverage_score": _clip(coverage),
            "temporal_score": _clip(temporal),
            "hallucination_score": _clip(hallucination),
        }

    def _compute_final_score(scores: Dict[str, float]) -> float:
        faithfulness = _clip(scores.get("faithfulness_score", 0.0))
        coverage = _clip(scores.get("coverage_score", 0.0))
        temporal = _clip(scores.get("temporal_score", 0.0))
        hallucination = _clip(scores.get("hallucination_score", 0.0))

        final_score = (
            0.4 * faithfulness
            + 0.25 * coverage
            + 0.25 * temporal
            - 0.3 * hallucination
        )
        return _clip(final_score)

    def _fn(videos, original_prompts, metadata=None):

        # decomposed_outputs = [x["decomposed_outputs"] for x in metadata]
        decomposed_outputs = [x["decomposed_llm_outputs"] for x in metadata]
        B = len(original_prompts)
        assert len(decomposed_outputs) == B, "len(decomposed_outputs) must equal len(original_prompts)"
        assert len(metadata) == B, "len(metadata) must equal len(original_prompts)"
        inner_group = metadata[0].get("inner_group", 1)
        actual_batch_size = B // inner_group

        scores = torch.zeros([actual_batch_size, 1], dtype=torch.float32)

        faithfulness_list = []
        coverage_list = []
        temporal_list = []
        hallucination_list = []
        raw_outputs = []

        for i in range(actual_batch_size):
            original_prompt = original_prompts[i * inner_group]
            decomposed_steps = decomposed_outputs[i * inner_group]

            num_votes = metadata[i].get("num_votes", 1)
            return_subscores = metadata[i].get("return_subscores", True)

            vote_scores = []
            last_raw_output = None

            for _ in range(num_votes):

                # judge_prompt = _build_judge_prompt(original_prompt, decomposed_steps)
                judge_prompt = _build_judge_single_promptv3(original_prompt, decomposed_steps)
                raw_output = call_llm_api(judge_prompt, max_retries=3)
                last_raw_output = raw_output
                # print(raw_output)
                try:
                    sub_scores = _safe_json_load(raw_output)
                except Exception:
                    sub_scores = _fallback_score(original_prompt, decomposed_steps)
                    last_raw_output = json.dumps(sub_scores, ensure_ascii=False)
                

                vote_scores.append({
                    "faithfulness_score": _clip(sub_scores.get("faithfulness_score", 0.0)),
                    "coverage_score": _clip(sub_scores.get("coverage_score", 0.0)),
                    "temporal_score": _clip(sub_scores.get("temporal_score", 0.0)),
                    "hallucination_score": _clip(sub_scores.get("hallucination_score", 0.0)),
                })

            # 多次投票平均
            avg_scores = {
                "faithfulness_score": sum(v["faithfulness_score"] for v in vote_scores) / num_votes,
                "coverage_score": sum(v["coverage_score"] for v in vote_scores) / num_votes,
                "temporal_score": sum(v["temporal_score"] for v in vote_scores) / num_votes,
                "hallucination_score": sum(v["hallucination_score"] for v in vote_scores) / num_votes,
            }

            final_score = _compute_final_score(avg_scores)
            scores[i, 0] = final_score

            faithfulness_list.append(avg_scores["faithfulness_score"])
            coverage_list.append(avg_scores["coverage_score"])
            temporal_list.append(avg_scores["temporal_score"])
            hallucination_list.append(avg_scores["hallucination_score"])
            raw_outputs.append(last_raw_output)

        info = {
            "outputs": raw_outputs,
        }

        # 默认返回子分数
        if any(m.get("return_subscores", True) for m in metadata):
            info["subscores"] = {
                "faithfulness_score": torch.tensor(faithfulness_list, dtype=torch.float32),
                "coverage_score": torch.tensor(coverage_list, dtype=torch.float32),
                "temporal_score": torch.tensor(temporal_list, dtype=torch.float32),
                "hallucination_score": torch.tensor(hallucination_list, dtype=torch.float32),
            }

        scores = scores.repeat(1, inner_group).reshape(-1)
        return scores, info

    return _fn
