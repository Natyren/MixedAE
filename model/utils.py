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
        flatten,
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
