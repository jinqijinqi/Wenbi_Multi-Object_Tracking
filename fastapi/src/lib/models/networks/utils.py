"""utils.py"""

"""
Ref: https://huggingface.co/transformers/_modules/transformers/modeling_utils.html
Author: Yimin Yang
Last revision date: Jan 18, 2022
Function: Some static functions for model building.
"""

import os
import random
from typing import List, Tuple
import numpy as np
import torch
from einops import repeat, rearrange
from torch import Tensor, nn


def expand_to_batch(tensor, desired_size):
    tile = desired_size // tensor.shape[0]
    return repeat(tensor, 'b ... -> (b tile) ...', tile=tile)


def init_random_seed(seed, gpu=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if gpu:
        torch.backends.cudnn.deterministic = True


def get_module_device(parameter: nn.Module):
    try:
        return next(parameter.parameters()).device
    except StopIteration:
        # For nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].device


def compute_mhsa(q, k, v, scale_factor=1, mask=None):
    # resulted shape will be: [batch, heads, tokens, tokens]
    scaled_dot_prod = torch.einsum('... i d , ... j d -> ... i j', q, k) * scale_factor

    if mask is not None:
        assert mask.shape == scaled_dot_prod.shape[2:]
        scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

    attention = torch.softmax(scaled_dot_prod, dim=-1)
    # calc result per head
    return torch.einsum('... i j , ... j d -> ... i d', attention, v)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None):
        """
        Implementation of multi-head attention layer of the original transformer model.
        einsum and einops.rearrange is used whenever possible
        Args:
            dim: token's dimension, i.e. word embedding vector size
            heads: the number of distinct representations to learn
            dim_head: the dim of the head. In general dim_head<dim.
            However, it may not necessary be (dim/heads)
        """
        super().__init__()
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads
        self.to_qvk = nn.Linear(dim, _dim * 3, bias=False)
        self.W_0 = nn.Linear(_dim, dim, bias=False)
        self.scale_factor = self.dim_head ** -0.5

    def forward(self, x, mask=None):
        assert x.dim() == 3
        qkv = self.to_qvk(x)  # [batch, tokens, dim*3*heads ]

        # decomposition to q,v,k and cast to tuple
        # the resulted shape before casting to tuple will be: [3, batch, heads, tokens, dim_head]
        q, k, v = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.heads))

        out = compute_mhsa(q, k, v, mask=mask, scale_factor=self.scale_factor)

        # re-compose: merge heads with dim_head
        out = rearrange(out, "b h t d -> b t (h d)")
        # Apply final linear transformation layer
        return self.W_0(out)


class TransformerBlock(nn.Module):
    """
    Vanilla transformer block from the original paper "Attention is all you need"
    Detailed analysis: https://theaisummer.com/transformer/
    """

    def __init__(self, dim, heads=8, dim_head=None,
                 dim_linear_block=1024, dropout=0.1, activation=nn.GELU,
                 mhsa=None, prenorm=False):
        """
        Args:
            dim: token's vector length
            heads: number of heads
            dim_head: if none dim/heads is used
            dim_linear_block: the inner projection dim
            dropout: probability of droppping values
            mhsa: if provided you can change the vanilla self-attention block
            prenorm: if the layer norm will be applied before the mhsa or after
        """
        super().__init__()
        self.mhsa = mhsa if mhsa is not None else MultiHeadSelfAttention(dim=dim, heads=heads, dim_head=dim_head)
        self.prenorm = prenorm
        self.drop = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)

        self.linear = nn.Sequential(
            nn.Linear(dim, dim_linear_block),
            activation(),  # nn.ReLU or nn.GELU
            nn.Dropout(dropout),
            nn.Linear(dim_linear_block, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        if self.prenorm:
            y = self.drop(self.mhsa(self.norm_1(x), mask)) + x
            out = self.linear(self.norm_2(y)) + y
        else:
            y = self.norm_1(self.drop(self.mhsa(x, mask)) + x)
            out = self.norm_2(self.linear(y) + y)
        return out


class TransformerEncoder(nn.Module):

    def __init__(self, dim, blocks=6, heads=8, dim_head=None, dim_linear_block=1024, dropout=0, prenorm=False):
        super().__init__()
        self.block_list = [TransformerBlock(dim, heads, dim_head,
                                            dim_linear_block, dropout, prenorm=prenorm) for _ in range(blocks)]
        self.layers = nn.ModuleList(self.block_list)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class ViT(nn.Module):
    def __init__(self, *,
                 img_dim,
                 in_channels=3,
                 patch_dim=16,
                 num_classes=1,
                 dim=512,
                 blocks=6,
                 heads=4,
                 dim_linear_block=1024,
                 dim_head=None,
                 dropout=0.1, transformer=None, classification=True):
        """
        Minimal re-implementation of ViT
        Args:
            img_dim: the spatial image size
            in_channels: number of img channels
            patch_dim: desired patch dim
            num_classes: classification task classes
            dim: the linear layer's dim to project the patches for MHSA
            blocks: number of transformer blocks
            heads: number of heads
            dim_linear_block: inner dim of the transformer linear block
            dim_head: dim head in case you want to define it. defaults to dim/heads
            dropout: for pos emb and transformer
            transformer: in case you want to provide another transformer implementation
            classification: creates an extra CLS token that we will index in the final classification layer
        """
        super().__init__()
        assert img_dim % patch_dim == 0, f'patch size {patch_dim} not divisible by img dim {img_dim}'
        self.p = patch_dim
        self.classification = classification
        # tokens = number of patches
        tokens = (img_dim // patch_dim) ** 2
        self.token_dim = in_channels * (patch_dim ** 2)
        self.dim = dim
        self.dim_head = (int(self.dim / heads)) if dim_head is None else dim_head

        # Projection and pos embeddings
        self.project_patches = nn.Linear(self.token_dim, self.dim)

        self.emb_dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.pos_emb1D = nn.Parameter(torch.randn(tokens + 1, self.dim))

        if self.classification:
            self.mlp_head = nn.Linear(self.dim, num_classes)

        if transformer is None:
            self.transformer = TransformerEncoder(self.dim, blocks=blocks, heads=heads,
                                                  dim_head=self.dim_head,
                                                  dim_linear_block=dim_linear_block,
                                                  dropout=dropout)
        else:
            self.transformer = transformer

    def forward(self, img, mask=None):
        # Create patches
        # from [batch, channels, h, w] to [batch, tokens , N], N=p*p*c , tokens = h/p *w/p
        img_patches = rearrange(img,
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.p, patch_y=self.p)

        batch_size, tokens, _ = img_patches.shape

        # project patches with linear layer + add pos emb
        img_patches = self.project_patches(img_patches)

        img_patches = torch.cat((expand_to_batch(self.cls_token, desired_size=batch_size), img_patches), dim=1)

        # add pos. embeddings. + dropout
        # indexing with the current batch's token length to support variable sequences
        img_patches = img_patches + self.pos_emb1D[:tokens + 1, :]
        patch_embeddings = self.emb_dropout(img_patches)

        # feed patch_embeddings and output of transformer. shape: [batch, tokens, dim]
        y = self.transformer(patch_embeddings, mask)

        # we index only the cls token for classification. nlp tricks :P
        return self.mlp_head(y[:, 0, :]) if self.classification else y[:, 1:, :]

    class SingleConv(nn.Module):
        """
        Double convolution block that keeps that spatial sizes the same
        """

        def __init__(self, in_ch, out_ch, norm_layer=None):
            super(SingleConv, self).__init__()

            if norm_layer is None:
                norm_layer = nn.BatchNorm2d

            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                norm_layer(out_ch),
                nn.ReLU(inplace=True))

        def forward(self, x):
            return self.conv(x)

    class DoubleConv(nn.Module):
        """
        Double convolution block that keeps that spatial sizes the same
        """

        def __init__(self, in_ch, out_ch, norm_layer=None):
            super(DoubleConv, self).__init__()
            self.conv = nn.Sequential(SingleConv(in_ch, out_ch, norm_layer),
                                      SingleConv(out_ch, out_ch, norm_layer))

        def forward(self, x):
            return self.conv(x)

    class DepthwiseSeparableConv(nn.Module):
        def __init__(self, in_channels, output_channels, kernels_per_layer=1):
            super(DepthwiseSeparableConv, self).__init__()
            self.depthwise = nn.Conv2d(in_channels, in_channels * kernels_per_layer, groups=in_channels, kernel_size=1)
            self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, output_channels, kernel_size=1)

        def forward(self, x):
            x = self.depthwise(x)
            x = self.pointwise(x)
            return x

    class DoubleConvDS(nn.Module):
        """(convolution => [BN] => ReLU) * 2"""

        def __init__(self, in_channels, out_channels, mid_channels=None, kernels_per_layer=1):
            super().__init__()
            if not mid_channels:
                mid_channels = out_channels
            self.double_conv = nn.Sequential(
                DepthwiseSeparableConv(in_channels, mid_channels, kernels_per_layer=kernels_per_layer),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                DepthwiseSeparableConv(mid_channels, out_channels, kernels_per_layer=kernels_per_layer),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            return self.double_conv(x)

    class UpDS(nn.Module):
        """Upscaling then double conv"""

        def __init__(self, in_channels, out_channels, bilinear=True, kernels_per_layer=1):
            super().__init__()

            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.conv = DoubleConvDS(in_channels, out_channels, in_channels // 2,
                                         kernels_per_layer=kernels_per_layer)
            else:
                self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
                self.conv = DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer)

        def forward(self, x1, x2=None):
            x = self.up(x1)
            if x2 is not None:
                x = torch.cat([x2, x], dim=1)
            return self.conv(x)

    class Flatten(nn.Module):
        def forward(self, x):
            return x.view(x.size(0), -1)

    class ChannelAttention(nn.Module):
        def __init__(self, input_channels, reduction_ratio=16):
            super(ChannelAttention, self).__init__()
            self.input_channels = input_channels
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            #  uses Convolutions instead of Linear
            self.MLP = nn.Sequential(
                Flatten(),
                nn.Linear(input_channels, input_channels // reduction_ratio),
                nn.ReLU(),
                nn.Linear(input_channels // reduction_ratio, input_channels)
            )

        def forward(self, x):
            # Take the input and apply average and max pooling
            avg_values = self.avg_pool(x)
            max_values = self.max_pool(x)
            out = self.MLP(avg_values) + self.MLP(max_values)
            scale = x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
            return scale

    class SpatialAttention(nn.Module):
        def __init__(self, kernel_size=7):
            super(SpatialAttention, self).__init__()
            assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
            padding = 3 if kernel_size == 7 else 1
            self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
            self.bn = nn.BatchNorm2d(1)

        def forward(self, x):
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            out = torch.cat([avg_out, max_out], dim=1)
            out = self.conv(out)
            out = self.bn(out)
            scale = x * torch.sigmoid(out)
            return scale

    class CBAM(nn.Module):
        def __init__(self, input_channels, reduction_ratio=16, kernel_size=7):
            super(CBAM, self).__init__()
            self.channel_att = ChannelAttention(input_channels, reduction_ratio=reduction_ratio)
            self.spatial_att = SpatialAttention(kernel_size=kernel_size)

        def forward(self, x):
            out = self.channel_att(x)
            out = self.spatial_att(out)
            return out

    def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=dilation, groups=groups, bias=False, dilation=dilation)

    def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
        """1x1 convolution"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    class Bottleneck(nn.Module):
        # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
        # while original implementation places the stride at the first 1x1 convolution(self.conv1)
        expansion = 1

        def __init__(self, inplanes, planes, stride=1, groups=1,
                     base_width=64, dilation=1, norm_layer=None):
            super(Bottleneck, self).__init__()

            if norm_layer is None:
                norm_layer = nn.BatchNorm2d

            if stride != 1 or inplanes != planes * self.expansion:
                self.downsample = nn.Sequential(
                    conv1x1(inplanes, planes * self.expansion, stride),
                    norm_layer(planes * self.expansion),
                )
            else:
                self.downsample = nn.Identity()

            width = int(planes * (base_width / 64.)) * groups

            self.conv1 = conv1x1(inplanes, width)
            self.bn1 = norm_layer(width)
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
            self.bn2 = norm_layer(width)
            self.conv3 = conv1x1(width, planes * self.expansion)
            self.bn3 = norm_layer(planes * self.expansion)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)
            identity = self.downsample(x)
            out += identity
            out = self.relu(out)

            return out