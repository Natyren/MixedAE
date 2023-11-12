# from functools import partial

import torch
import torch.nn as nn

# from .utils import get_2d_sincos_pos_embed
from .modeling import PatchEmbed, Block
from .utils import mixing


class MixedMaskedAutoencoderViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )
        self.segment_embed = nn.Embedding(
            4, embed_dim
        )  # TODO change 4 to custom mixing param

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

    def shuffling(self, x, n_splits=4):
        mixed = [mixing(tnsr) for tnsr in torch.split(x, n_splits)]
        x_tensors = torch.cat([tnsr[0] for tnsr in mixed])
        idxes = torch.cat([tnsr[1] for tnsr in mixed])
        return x_tensors, idxes

    def forward_encoder(self, x):
        x = self.patch_embed(x)

        x = x + self.pos_embed[:, 1:, :]
        x = x + self.segment_embed(
            torch.tensor(
                [i % 4 for i in range(x.shape[0])]
            )  # TODO change 4 to custom mixing param
        ).unsqueeze(1)

        x, ids = self.shuffling(x)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, ids
