from functools import partial
import collections

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        proj,
        in_channels,
        embed_dim,
        norm_layer,
        bias,
    ):
        super().__init__()
        assert isinstance(img_size, collections.abc.Iterable) or isinstance(
            img_size, int
        ), "Please Provide img_size in iterable or int format"
        self.img_size = (
            (img_size, img_size)
            if not isinstance(img_size, collections.abc.Iterable)
            else patch_size
        )

        assert isinstance(patch_size, collections.abs.Iterable) or isinstance(
            patch_size, int
        ), "Please provide patch_size in iterable or int format"
        self.patch_size = (
            (patch_size, patch_size)
            if not isinstance(patch_size, collections.abc.Iterable)
            else patch_size
        )

        if proj.lower() == "linear":
            raise NotImplementedError(
                "This type of projection is not implemented now"
            )
        elif proj.lower() == "conv":
            self.proj = nn.Conv2d(
                in_channels,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                bias=bias,
            )

        if norm_layer:
            if isinstance(norm_layer, nn.Module):
                self.norm_layer = norm_layer
            else:
                raise ValueError(
                    "Please provide norm_layer in torch.nn.Module format"
                )
        else:
            self.norm_layer = nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape
        if self.img_size is not None:
            assert H == self.img_size[0]
            assert W == self.img_size[1]
        x = self.proj(x)
        x = self.norm_layer(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HomoAttention(nn.Module):
    def __init__(
        self,
        dim,
        topk=5,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.topk = topk
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qk = (
            self.qk(x)
            .reshape(B, N, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k = qk.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # implementation of topk operation

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    pass
