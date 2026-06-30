from flow_grpo.video_pickscore_scorer import PickScoreVideoScorer, load_video_mp4
import pandas as pd
import os
import torch
from tqdm import tqdm 
import json
from copy import deepcopy
import ast

def remove_overlap_between_steps(step_data):
    """
    删除每个 step prompt 中，与下一段开头重复的部分。

    输入格式:
    [
        {"step_index": 0, "prompt": "..."},
        ...
    ]
    """

    results = deepcopy(step_data)

    for i in range(len(results) - 1):
        curr_prompt = results[i]["prompt"].strip()
        next_prompt = results[i + 1]["prompt"].strip()

        # 找最长公共 overlap：
        # 当前 prompt 的结尾 == 下一 prompt 的开头
        max_overlap = ""

        max_len = min(len(curr_prompt), len(next_prompt))

        for k in range(1, max_len + 1):
            if curr_prompt[-k:] == next_prompt[:k]:
                max_overlap = curr_prompt[-k:]

        # 删除 overlap
        if max_overlap:
            results[i]["prompt"] = curr_prompt[:-len(max_overlap)].strip()

    return results


reward_score_fn = PickScoreVideoScorer(device="cuda", dtype=torch.float32)

test_root_paths = [
    "/path/to/output/run1",
    "/path/to/output/run2",
]

for root_path in test_root_paths:
    csv_path = os.path.join(root_path, "results.csv")
    out_path = os.path.join(root_path, "pickscore_results.csv")
    if os.path.exists(out_path):
        continue

    if "simple100" in root_path:
        metadata = [{"eval_sample_frames": 6, "gap_frame": 12, "num_votes": 1, "train_frames": 24}]
    else:
        metadata = [{"eval_sample_frames": 10, "gap_frame": 12, "num_votes": 1, "train_frames": 24}]

    df = pd.read_csv(csv_path)

    # 创建输出文件，写入包含所有列的表头
    if not os.path.exists(out_path):
        header_df = pd.DataFrame(columns=list(df.columns) + ["score", "outputs"])
        header_df.to_csv(out_path, index=False)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
        video_paths = [row["video_path"]]
        prompts = [row["prompt"]]

        videos = [load_video_mp4(p, num_frames=8) for p in video_paths]

        score = reward_score_fn(prompts, videos, metadata)
        
        # 将tensor转换为float
        score_float = score.item() if hasattr(score, 'item') else float(score)
        # output_text = score_metadata["outputs"][0]
        # json_str = json.dumps(output_text, ensure_ascii=False)

        # 创建当前行的结果
        result_row = pd.DataFrame([{
            **row.to_dict(),
            "score": score_float,
            # "outputs": json_str
        }])
        
        # 追加写入到CSV文件
        result_row.to_csv(out_path, mode='a', header=False, index=False)

    # 计算平均分
    processed_df = pd.read_csv(out_path)
    mean_score = processed_df["score"].mean()
    print(root_path)
    print(f"mean_score is {mean_score}| total sample number {len(processed_df)}")
