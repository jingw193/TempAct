from flow_grpo.gemini_reward import gemini_video_score
import pandas as pd
import os
import torch
from tqdm import tqdm
import json
from tools.metric.compute_gemini_score import main as compute_gemini_score

reward_score_fn = gemini_video_score("cuda")
metadata = [{"eval_sample_frames": 6}]

root_path = "/path/to/output/run"
csv_path = os.path.join(root_path, "inference_results.csv")
out_path = os.path.join(root_path, "gemini_results.csv")

df = pd.read_csv(csv_path)

# =========================
# 1. 读取已有结果（用于断点续跑）
# =========================
if os.path.exists(out_path):
    processed_df = pd.read_csv(out_path)
    if "video" in processed_df.columns:
        done_set = set(processed_df["video"].tolist())
    else:
        done_set = set()
    print(f"[Resume] Found existing results: {len(done_set)} processed samples")
else:
    processed_df = None
    done_set = set()
    # 创建文件 + header
    header_df = pd.DataFrame(columns=list(df.columns) + ["score", "outputs"])
    header_df.to_csv(out_path, index=False)

# =========================
# 2. 主循环（跳过已完成）
# =========================
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):

    video_path = row["video"]
    # video_path = row["video_path"]

    # skip 已处理
    if video_path in done_set:
        continue

    video_paths = [video_path]
    prompts = [row["caption"]]

    try:
        score, score_metadata = reward_score_fn(video_paths, prompts, metadata)

        score_float = score.item() if hasattr(score, 'item') else float(score)
        output_text = score_metadata["outputs"][0]
        json_str = json.dumps(output_text, ensure_ascii=False)

    except Exception as e:
        print(f"[Error] idx={idx}, video={video_path}, err={e}")
        score_float = -1.0
        json_str = json.dumps({"error": str(e)}, ensure_ascii=False)

    result_row = pd.DataFrame([{
        **row.to_dict(),
        "score": score_float,
        "outputs": json_str
    }])

    result_row.to_csv(out_path, mode='a', header=False, index=False)

# =========================
compute_gemini_score(out_path, end_index=100, sup_test=True)
# 3. 统计
# =========================
processed_df = pd.read_csv(out_path)
mean_score = processed_df["score"].mean()

print(root_path)
print(f"mean_score is {mean_score}| total sample number {len(processed_df)}")
