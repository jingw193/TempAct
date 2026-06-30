from PIL import Image
import io
import numpy as np
import torch
from collections import defaultdict
from flow_grpo.gemini_reward import gemini_video_score
from flow_grpo.qwen3vl_video_reward import qwen3vl_video_score, qwen3vl_video_local_score
from flow_grpo.llm_judge_score import decomposition_reward_score

def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew / 500, meta

    return _fn


def aesthetic_score(device):
    from flow_grpo.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


def clip_score(device):
    from flow_grpo.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device)

    def _fn(images, prompts, metadata):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8) / 255.0
        scores = scorer(images, prompts)
        return scores, {}

    return _fn


def hpsv2_score(device):
    from flow_grpo.hpsv2_scorer import HPSv2Scorer

    scorer = HPSv2Scorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8) / 255.0
        scores = scorer(images, prompts)
        return scores, {}

    return _fn


def pickscore_score(device):
    from flow_grpo.pickscore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def videopickscore_score(device):
    from flow_grpo.video_pickscore_scorer import PickScoreVideoScorer

    scorer = PickScoreVideoScorer(dtype=torch.float32, device=device)

    def _fn(videos, prompts, metadata):
        if isinstance(videos, torch.Tensor):
            videos = (videos * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            videos = videos.transpose(0, 1, 3, 4, 2)  # BFCHW -> NFHWC
            B, F, H, W, C = videos.shape
            num_frames = 8
            idx = torch.linspace(0, F - 1, num_frames).long()
            videos = videos[:, idx]
            images = []
            for video in videos:
                images_list = [Image.fromarray(image) for image in video]
                images.append(images_list)
            # images = [Image.fromarray(image) for image in images]
        scores = scorer(prompts, images, metadata)
        return scores, {}

    return _fn


def videopickscore_local_score(device):
    from flow_grpo.video_pickscore_scorer import PickScoreVideoScorer

    scorer = PickScoreVideoScorer(dtype=torch.float32, device=device)

    def _fn(videos, prompts, metadata):
        decomposed_outputs = [x["decomposed_llm_outputs"] for x in metadata]
        train_frames = metadata[0].get("train_frames", 15)
        eval_frames_region = metadata[0].get("gap_frame", 6) * 4
        start_frame = (train_frames - 1) * 4 + 1

        prompt_step_idx = train_frames // metadata[0].get("gap_frame", 6)
        step_prompt = [decomposed_output[prompt_step_idx]["prompt"] for decomposed_output in decomposed_outputs]

        if isinstance(videos, torch.Tensor):
            videos = (videos * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            videos = videos.transpose(0, 1, 3, 4, 2)  # BFCHW -> NFHWC
            B, F, H, W, C = videos.shape
            num_frames = 8

            idx = torch.linspace(start_frame, start_frame + eval_frames_region - 1, num_frames).long()
            videos = videos[:, idx]
            images = []
            for video in videos:
                images_list = [Image.fromarray(image) for image in video]
                images.append(images_list)
            # images = [Image.fromarray(image) for image in images]
        scores = scorer(step_prompt, images, metadata)
        return scores, {}

    return _fn


def imagereward_score(device):
    from flow_grpo.imagereward_scorer import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn


def geneval_score(device):
    from flow_grpo.gen_eval import load_geneval

    batch_size = 64
    compute_geneval = load_geneval(device)

    def _fn(images, prompts, metadatas, only_strict):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadatas_batched = np.array_split(metadatas, np.ceil(len(metadatas) / batch_size))
        all_scores = []
        all_rewards = []
        all_strict_rewards = []
        all_group_strict_rewards = []
        all_group_rewards = []
        for image_batch, metadata_batched in zip(images_batched, metadatas_batched):
            pil_images = [Image.fromarray(image) for image in image_batch]

            data = {
                "images": pil_images,
                "metadatas": list(metadata_batched),
                "only_strict": only_strict,
            }
            scores, rewards, strict_rewards, group_rewards, group_strict_rewards = compute_geneval(**data)

            all_scores += scores
            all_rewards += rewards
            all_strict_rewards += strict_rewards
            all_group_strict_rewards.append(group_strict_rewards)
            all_group_rewards.append(group_rewards)
        all_group_strict_rewards_dict = defaultdict(list)
        all_group_rewards_dict = defaultdict(list)
        for current_dict in all_group_strict_rewards:
            for key, value in current_dict.items():
                all_group_strict_rewards_dict[key].extend(value)
        all_group_strict_rewards_dict = dict(all_group_strict_rewards_dict)

        for current_dict in all_group_rewards:
            for key, value in current_dict.items():
                all_group_rewards_dict[key].extend(value)
        all_group_rewards_dict = dict(all_group_rewards_dict)

        return all_scores, all_rewards, all_strict_rewards, all_group_rewards_dict, all_group_strict_rewards_dict

    return _fn

def videoalign_score(device):
    from flow_grpo.video_align_score import VideoAlignScorer

    reward_type = "TA"
    # reward_type selects the scoring dimension: "Overall", "VQ" (visual quality),
    # "MQ" (motion quality, grayscale-sensitive), or "TA" (text alignment)
    scorer = VideoAlignScorer(device=device, dtype=torch.bfloat16, reward_type=reward_type, use_grayscale=True)

    def _fn(videos, prompts, metadata=None):
        if not isinstance(videos, torch.Tensor):
            videos = torch.from_numpy(videos).permute(0, 1, 4, 2, 3)
        
        if videos.dtype != torch.uint8:
            videos = (videos * 255).round().clamp(0, 255).to(torch.uint8)

        scores_tensor = scorer(list(videos), prompts)
        return scores_tensor, {}

    return _fn


def ocr_score(device):
    from flow_grpo.ocr import OcrScorer

    scorer = OcrScorer()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn

def acc_error_score(device):
    from flow_grpo.accmulate_error import AccErrorScore

    scorer = AccErrorScore()

    def _fn(latents, prompts, metadata):
        # if isinstance(images, torch.Tensor):
        #     images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        #     images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        scores = scorer(latents, metadata)
        # change tensor to list
        return scores, {}

    return _fn

def unifiedreward_score_sglang(device):
    import asyncio
    from openai import AsyncOpenAI
    import base64
    from io import BytesIO
    import re

    def pil_image_to_base64(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        return base64_qwen

    def _extract_scores(text_outputs):
        scores = []
        pattern = r"Final Score:\s*([1-5](?:\.\d+)?)"
        for text in text_outputs:
            match = re.search(pattern, text)
            if match:
                try:
                    scores.append(float(match.group(1)))
                except ValueError:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return scores

    client = AsyncOpenAI(base_url="http://127.0.0.1:17140/v1", api_key="flowgrpo")

    async def evaluate_image(prompt, image):
        question = f"<image>\nYou are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nBased on the above criteria, assign a score from 1 to 5 after 'Final Score:'.\nYour task is provided as follows:\nText Caption: [{prompt}]"
        images_base64 = pil_image_to_base64(image)
        response = await client.chat.completions.create(
            model="UnifiedReward-7b-v1.5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": images_base64},
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    async def evaluate_batch_image(images, prompts):
        tasks = [evaluate_image(prompt, img) for prompt, img in zip(prompts, images)]
        results = await asyncio.gather(*tasks)
        return results

    def _fn(images, prompts, metadata):
        # 处理Tensor类型转换
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        # 转换为PIL Image并调整尺寸
        images = [Image.fromarray(image).resize((512, 512)) for image in images]

        # 执行异步批量评估
        text_outputs = asyncio.run(evaluate_batch_image(images, prompts))
        score = _extract_scores(text_outputs)
        score = [sc / 5.0 for sc in score]
        return score, {}

    return _fn


def multi_score(device, score_dict):
    score_functions = {
        "ocr": ocr_score,
        "imagereward": imagereward_score,
        "pickscore": pickscore_score,
        "videopickscore_score": videopickscore_score,
        "videopickscore_local_score": videopickscore_local_score,
        "aesthetic": aesthetic_score,
        "jpeg_compressibility": jpeg_compressibility,
        "unifiedreward": unifiedreward_score_sglang,
        "geneval": geneval_score,
        "clipscore": clip_score,
        "hpsv2": hpsv2_score,
        "accerrorscore": acc_error_score,
        "gemini_score": gemini_video_score,
        "videoalign_score": videoalign_score,
        "qwen3vl_video_score": qwen3vl_video_score,
        "qwen3vl_local_score": qwen3vl_video_local_score,
        "llm_judge_score": decomposition_reward_score
    }
    score_fns = {}
    for score_name, weight in score_dict.items():
        score_fns[score_name] = (
            score_functions[score_name](device)
            if "device" in score_functions[score_name].__code__.co_varnames
            else score_functions[score_name]()
        )

    # only_strict is only for geneval. During training, only the strict reward is needed, and non-strict rewards don't need to be computed, reducing reward calculation time.
    def _fn(images, prompts, metadata, only_strict=True):
        total_scores = None
        score_details = {}

        for score_name, weight in score_dict.items():
            if score_name == "geneval":
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards = score_fns[score_name](
                    images, prompts, metadata, only_strict
                )
                score_details["accuracy"] = rewards
                score_details["strict_accuracy"] = strict_rewards
                for key, value in group_strict_rewards.items():
                    score_details[f"{key}_strict_accuracy"] = value
                for key, value in group_rewards.items():
                    score_details[f"{key}_accuracy"] = value
            else:
                scores, rewards = score_fns[score_name](images, prompts, metadata)
            score_details[score_name] = scores

            if score_name not in ["accerrorscore", "gemini_score", "videoalign_score", "videopickscore_score", "qwen3vl_video_score", "llm_judge_score", "qwen3vl_local_score", "videopickscore_local_score"]:
                weighted_scores = [weight * score for score in scores]

                if not total_scores:
                    total_scores = weighted_scores
                else:
                    total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]
            else:
                weighted_scores = weight * scores
                if total_scores is None:
                    total_scores = weighted_scores
                else:
                    total_scores += weighted_scores

        score_details["avg"] = total_scores
        return score_details, {}

    return _fn
