import torch
import torch.nn as nn
import numpy as np


class HomoContrastive(nn.Module):
    def __init__(
        self, temperature=0.07, contrast_mode="all", base_temperature=0.07
    ):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        total_loss = 0
        for _labels, item in zip(labels, features):
            labels = _labels[:, 0].view(-1, 1)

            mask = torch.eq(labels, labels.T).float().to(device)

            contrast_count = item.shape[0]
            contrast_feature = item
            if self.contrast_mode == "one":
                anchor_feature = item[:, 0]
                anchor_count = 1
            elif self.contrast_mode == "all":
                anchor_feature = contrast_feature
                anchor_count = contrast_count
            else:
                raise ValueError("Unknown mode: {}".format(self.contrast_mode))
            # compute logits
            anchor_dot_contrast = torch.div(
                torch.matmul(anchor_feature, contrast_feature.T),
                self.temperature,
            )
            # for numerical stability
            logits_min, _ = torch.min(anchor_dot_contrast, dim=1, keepdim=True)
            logits_std = torch.std(anchor_dot_contrast, dim=1, keepdim=True)
            logits = (anchor_dot_contrast - logits_min.detach()) / logits_std

            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(anchor_count).view(-1, 1).to(device),
                0,
            )
            mask = mask * logits_mask
            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

            # loss
            loss = (
                -(self.temperature / self.base_temperature) * mean_log_prob_pos
            )
            loss = loss.view(anchor_count).mean()
            total_loss += loss

        return total_loss


def mixing(
    a,
):  # a.shape = [B, (img_size/patch_size)**2, hidden_dim] where B = 2 or 4
    idx = torch.cat(
        [
            torch.randperm(a.size(-3)).unsqueeze(-1).repeat(1, a.size(-1))
            for _ in range(a.size(-2))
        ],
        axis=1,
    ).reshape(a.shape)
    idx = idx.to(a.device)
    mixed = torch.gather(a, dim=0, index=idx)
    return mixed, idx


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0]
    )  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1]
    )  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
