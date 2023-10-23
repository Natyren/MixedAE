from functools import partial
import collections

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        proj="conv",
        in_channels=3,
        embed_dim=768,
        norm_layer=nn.LayerNorm,
        bias=True,
    ):
        super().__init__()
        assert isinstance(img_size, collections.abc.Iterable) or isinstance(
            img_size, int
        ), "Please Provide img_size in iterable or int format"
        self.img_size = (
            (img_size, img_size)
            if not isinstance(img_size, collections.abc.Iterable)
            else img_size
        )

        assert isinstance(patch_size, collections.abc.Iterable) or isinstance(
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
            if issubclass(norm_layer, nn.Module):
                self.norm_layer = norm_layer(embed_dim)
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
        x = x.flatten(2).transpose(1, 2)
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
        topk=0.25,
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

        self.topk = int(topk * self.head_dim)
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.num_heads * self.topk, dim)
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
        attn = torch.topk(attn, k=self.topk, dim=-1)[0]
        attn = attn.softmax(dim=-1)
        x = self.attn_drop(attn)

        x = x.transpose(1, 2).reshape(B, N, self.num_heads * self.topk)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = (bias, bias)
        drop_probs = (drop, drop)
        linear_layer = (
            partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        )

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features)
            if norm_layer is not None
            else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        homo=True,
        topk=0.25,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0.0,
        attn_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if homo:
            self.attn = HomoAttention(
                dim,
                topk=topk,
                num_heads=num_heads,
                qk_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer,
            )
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer,
            )

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
