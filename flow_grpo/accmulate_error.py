import torch
from typing import List, Dict

import torch
from typing import List, Dict, Tuple


class AccErrorScore:

    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        reduction: str = "mean",
        use_mse: bool = True,
        reward_scale: float = 5.0,
    ):
        self.device = device
        self.reduction = reduction
        self.use_mse = use_mse
        self.reward_scale = reward_scale

    def _reduce(self, x):
        if self.reduction == "mean":
            return x.mean(dim=(-1, -2, -3, -4))
        else:
            return x.sum(dim=(-1, -2, -3, -4))

    @torch.no_grad()
    def __call__(
        self,
        tuple_latents: Tuple[torch.Tensor, torch.Tensor],
        metadata: List[Dict],
    ) -> torch.Tensor:

        pred_latents, prefix_latents, _ = tuple_latents

        assert pred_latents.dim() == 6, f"Expected [B,G,3,C,H,W], got {pred_latents.shape}"
        assert prefix_latents.dim() == 5, f"Expected [B,pre_T,C,H,W], got {prefix_latents.shape}"

        B, G, T, C, H, W = pred_latents.shape
        assert T == 3
        k = 3

        rewards = torch.zeros(B, G, device=pred_latents.device)

        for b in range(B):

            latent_path = metadata[b]["latent_path"]

            # GT frames: T ... T+3
            gt_latent = torch.load(
                latent_path,
                map_location=self.device
            )["video_latent"][0]

            gt_T = gt_latent[-(T + k): -T]      # frame T # [k, C, H, W]
            gt_future = gt_latent[-T:]      # frames T+1..T+3.   # [T, C, H, W]

            gt_T = gt_T.to(pred_latents.device)
            gt_future = gt_future.to(pred_latents.device)

            pred = pred_latents[b]          # [G,T,C,H,W]
            prefix = prefix_latents[b, -k:]  # [k, C, H, W]

            # expand GT for group
            gt_future = gt_future.unsqueeze(0).expand(G, -1, -1, -1, -1)
            gt_T_expand = gt_T.unsqueeze(0).expand(G, -1, -1, -1, -1)

            # ---------- err0 ----------
            diff0 = prefix.unsqueeze(0) - gt_T_expand

            if self.use_mse:
                err0 = diff0.pow(2)
            else:
                err0 = diff0.abs()

            err0 = self._reduce(err0)   # [G]

            # ---------- err3 ----------
            diff = pred - gt_future

            if self.use_mse:
                err = diff.pow(2)
            else:
                err = diff.abs()

            err = self._reduce(err)     # [G]

            # err3 = err[:, -1]           # final frame error

            # ---------- drift reward ----------
            reward = -(err - err0)
            reward = torch.clamp(reward, max=0)

            reward = reward * self.reward_scale

            rewards[b] = reward

        return rewards