from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
import os
from decord import VideoReader, cpu

class PickScoreVideoScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32, frame_pool="mean"):
        super().__init__()
        processor_path = "/path/to/public_models/laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_path = "/path/to/public_models/yuvalkirstain/PickScore_v1"
        self.device = device
        self.dtype = dtype
        self.frame_pool = frame_pool

        self.processor = AutoProcessor.from_pretrained(processor_path)
        self.model = AutoModel.from_pretrained(model_path).eval().to(device)
        self.model = self.model.to(dtype=dtype)

    @torch.no_grad()
    def encode_video(self, video, local_sample_frames=4):  
        """
        video: Tensor [F, C, H, W]
        return: Tensor [D]
        """

        # F = video.shape[0]

        # [F, C, H, W] -> list[PIL or tensor frame]
        # processor 支持 list frame
        image_inputs = self.processor(
            images=video,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )            

        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}

        frame_embs = self.model.get_image_features(**image_inputs)
        frame_embs = frame_embs / frame_embs.norm(p=2, dim=-1, keepdim=True)

        # temporal aggregation
        if self.frame_pool == "mean":
            video_emb = frame_embs.mean(dim=0)
            local_video_emb = frame_embs[local_sample_frames:].mean(dim=0)
        elif self.frame_pool == "max":
            video_emb = frame_embs.max(dim=0).values
            local_video_emb = frame_embs[local_sample_frames:].max(dim=0).values
        else:
            raise ValueError("frame_pool must be mean or max")

        video_emb = video_emb / video_emb.norm(p=2, dim=-1, keepdim=True)
        local_video_emb = local_video_emb / local_video_emb.norm(p=2, dim=-1, keepdim=True)
        return video_emb, local_video_emb

    @torch.no_grad()
    def __call__(self, prompt, videos, metadata):
        """
        videos: Tensor [B, F, C, H, W]
        prompt: List[str] or str
        """
        try:
            B = videos.shape[0]
        except:
            B = len(videos)

        if isinstance(prompt, str):
            prompt = [prompt] * B

        # text encode
        text_inputs = self.processor(
            text=prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )

        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)

        # video encode
        video_embs = []
        local_video_embs = []
        train_frames_ratio = metadata[0].get("train_frames_ratio", 0.5)
        num_sample_frames = 8
        local_sample_frames = int(num_sample_frames * train_frames_ratio)
        # start_frame = 
        for b in range(B):
            video = videos[b]  # [F, C, H, W]

            # 🔥 frame sampling
            if isinstance(video, torch.Tensor):
                F = video.shape[0]
                idx = torch.linspace(0, F - 1, num_sample_frames).long()
                video = video[idx].to(torch.float32)
                video = video_tensor_to_pil(video)

            global_video_emb, local_video_emb = self.encode_video(video, local_sample_frames)
            video_embs.append(global_video_emb)
            local_video_embs.append(local_video_emb)
        video_embs = torch.stack(video_embs, dim=0)
        local_video_embs = torch.stack(local_video_embs, dim=0)
        # similarity
        logit_scale = self.model.logit_scale.exp()
        scores = logit_scale * (text_embs * video_embs).sum(dim=-1)
        local_scores = logit_scale * (text_embs * local_video_embs).sum(dim=-1)

        # normalize (keep consistent with your image reward)
        scores = scores / 26.0
        local_scores = local_scores / 26.0

        if "train_frames_ratio" in metadata[0].keys():
            return torch.stack([scores, local_scores], dim=-1)
        else:
            return scores.cpu()


def video_tensor_to_pil(video):
    """
    video: [F, C, H, W], float32
           range: [0,1] 或 [-1,1]
    return: List[PIL.Image]
    """

    video = video.detach().cpu()

    # 🔥 处理 range
    if video.min() < 0:
        video = (video + 1) / 2  # [-1,1] → [0,1]

    video = video.clamp(0, 1)

    pil_frames = []
    for frame in video:  # [C, H, W]
        frame = (frame * 255).permute(1, 2, 0).numpy().astype("uint8")
        pil_frames.append(Image.fromarray(frame))

    return pil_frames

def load_video_mp4(path, num_frames=8):
    """
    path: mp4 文件路径
    return: List[PIL.Image]
    """

    vr = VideoReader(path, ctx=cpu(0))
    total_frames = len(vr)

    # 🔥 均匀抽帧（推荐）
    idx = torch.linspace(0, total_frames - 1, num_frames).long().tolist()

    frames = vr.get_batch(idx)  # [num_frames, H, W, C]
    frames = frames.asnumpy()

    pil_frames = [Image.fromarray(frame) for frame in frames]
    return pil_frames
