import pandas as pd
import json
import numpy as np
import re
import os
from flow_grpo.gemini_reward import gemini_video_score

def parse_json_from_text(text):
    if pd.isna(text):
        return None

    text = str(text)

    # 1. 去 markdown
    text = text.replace("```json", "")
    text = text.replace("```", "")

    # 2. 有时会有 log / traceback，截断
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        return None

    json_str = text[start:end+1]

    # 3. 尝试直接 parse
    try:
        return json.loads(json_str)
    except:
        pass

    # 4. fallback：处理双重转义
    try:
        json_str = json_str.encode().decode("unicode_escape")
        return json.loads(json_str)
    except:
        return None

def main(csv_path, end_index=100, save_path=None, sup_test=False):
    reward_score_fn = gemini_video_score("cuda")
    metadata = [{"eval_sample_frames": 6}]

    df = pd.read_csv(csv_path)
    if save_path is None:
        csv_dir = os.path.dirname(csv_path)
        save_path = os.path.join(csv_dir, "realgemini3_flash_score_results.json")

    temporal_list = []
    physical_list = []
    visual_list = []
    prompt_list = []
    final_list = []
    combined_list = []

    results = []

    cnt = 0
    fail = 0

    for index, row in df.iterrows():
        if index > end_index:
            break

        data = parse_json_from_text(row.get("outputs", None))
        if data is None:
        
            if "gemini" not in csv_path:
                fail += 1
                continue
            if not sup_test:
                fail += 1
                continue
            video_paths = [row["video"]]
            prompts = [row["caption"]]
            try:
                score, score_metadata = reward_score_fn(video_paths, prompts, metadata)

                score_float = score.item() if hasattr(score, 'item') else float(score)
                output_text = score_metadata["outputs"][0]
                json_str = json.dumps(output_text, ensure_ascii=False)
                df.loc[index, "score"] = score_float
                df.loc[index, "outputs"] = json_str
                data = parse_json_from_text(json_str)
                df.to_csv(csv_path, index=False)
                frame_scores = np.array(data.get("frame_scores", []), dtype=np.float32) / 10.0

            except Exception as e:
                print(f"Error computing gemini score for index {index}: {e}")
                fail += 1
                continue

        frame_scores = np.array(data.get("frame_scores", []), dtype=np.float32) / 10.0
        global_score = float(data.get("final_score", 0)) / 10.0

        temporal_order_score = float(data.get("temporal_order_score", 0)) / 10.0
        physical_plausibility_score = float(data.get("physical_plausibility_score", 0)) / 10.0
        visual_consistency_score = float(data.get("visual_consistency_score", 0)) / 10.0
        prompt_alignment_score = float(data.get("prompt_alignment_score", 0)) / 10.0

        temporal_list.append(temporal_order_score)
        physical_list.append(physical_plausibility_score)
        visual_list.append(visual_consistency_score)
        prompt_list.append(prompt_alignment_score)
        final_list.append(global_score)

        alpha = 0.2
        beta = 0.3
        beta_1 = 0.2
        beta_2 = 0.1
        beta_3 = 0.1
        beta_4 = 0.1

        combined = (
            alpha * frame_scores.mean() +
            beta * global_score +
            beta_1 * temporal_order_score +
            beta_2 * physical_plausibility_score +
            beta_3 * visual_consistency_score +
            beta_4 * prompt_alignment_score
        )

        combined_list.append(combined)

        # ===== 保存每条样本 =====
        results.append({
            "index": index,
            "temporal_order_score": temporal_order_score,
            "physical_plausibility_score": physical_plausibility_score,
            "visual_consistency_score": visual_consistency_score,
            "prompt_alignment_score": prompt_alignment_score,
            "final_score": global_score,
            "combined_score": float(combined),
            "frame_score_mean": float(frame_scores.mean()) if len(frame_scores) > 0 else None,
        })

        cnt += 1

    # ===== 汇总 =====
    summary = {
        "parsed": cnt,
        "failed": fail,
        "avg_temporal_order_score": float(np.mean(temporal_list)) if temporal_list else 0,
        "avg_physical_plausibility_score": float(np.mean(physical_list)) if physical_list else 0,
        "avg_visual_consistency_score": float(np.mean(visual_list)) if visual_list else 0,
        "avg_prompt_alignment_score": float(np.mean(prompt_list)) if prompt_list else 0,
        "avg_final_score": float(np.mean(final_list)) if final_list else 0,
        "avg_combined_score": float(np.mean(combined_list)) if combined_list else 0,
    }

    output = {
        "summary": summary,
        "samples": results
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("parsed:", cnt)
    print("failed:", fail)
    print("===== AVERAGE METRICS =====")
    for k, v in summary.items():
        print(k, v)

    print(f"\nSaved to: {save_path}")


if __name__ == "__main__":
    csv_path = "csv_path"
    main(csv_path, end_index=100, sup_test=True)
