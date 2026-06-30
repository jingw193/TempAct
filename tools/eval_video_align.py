
from flow_grpo.gemini_reward import gemini_video_score
from flow_grpo.rewards import videoalign_score
from flow_grpo.video_align_score import VideoAlignScorer
import pandas as pd
import os
import torch
import torchvision
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reward_score_fn = videoalign_score(device=device)
metadata = [{"eval_sample_frames": 6}]

root_path = "/path/to/output/run"
csv_path = os.path.join(root_path, "inference_results.csv")
out_path = os.path.join(root_path, "videoalign_ta.csv")

df = pd.read_csv(csv_path)

scores = []
outputs = []
for idx, row in df.iterrows():
    print("idx", idx)
    video_path = row["video"]
    prompts = [row["caption"]]
    
    video_frames, _, _ = torchvision.io.read_video(video_path, pts_unit='sec', output_format='THWC')
    # [F, H, W, C] -> [F, C, H, W]
    video_frames = video_frames.permute(0, 3, 1, 2)
    # 添加 batch 维度
    videos = video_frames.unsqueeze(0).float() / 255.0
    score, _ = reward_score_fn(videos, prompts, metadata)
    scores.append(score[0].cpu().float().detach().numpy())

mean = np.mean(scores, axis=0)
print(mean)
df["video_align_ta_score"] = scores

df.to_csv(out_path, index=False)