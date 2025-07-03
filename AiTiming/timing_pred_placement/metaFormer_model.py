import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
from functools import partial
import os


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


class LayerNormGeneral(nn.Module):
    r""" General LayerNorm for different situations.

    Args:
        affine_shape (int, list or tuple): The shape of affine weight and bias.
            Usually the affine_shape=C, but in some implementation, like torch.nn.LayerNorm,
            the affine_shape is the same as normalized_shape. To adapt to different situations,
            we offer this argument here.
        normalized_dim (tuple or list): Which dims to compute mean and variance. 
        scale (bool): Flag to indicate whether to use scale or not.
        bias (bool): Flag to indicate whether to use scale or not.

        We give several examples to show how to specify the arguments.

        LayerNorm (https://arxiv.org/abs/1607.06450):
            For input shape of (B, *, C) like (B, N, C) or (B, H, W, C),
                affine_shape=C, normalized_dim=(-1, ), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(1, ), scale=True, bias=True.

        Modified LayerNorm (https://arxiv.org/abs/2111.11418)
            that is idental to partial(torch.nn.GroupNorm, num_groups=1):
            For input shape of (B, N, C),
                affine_shape=C, normalized_dim=(1, 2), scale=True, bias=True;
            For input shape of (B, H, W, C),
                affine_shape=C, normalized_dim=(1, 2, 3), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(2, 3), scale=True, bias=True.

        For the several metaformer baselines,
            IdentityFormer, RandFormer and PoolFormerV2 utilize Modified LayerNorm without bias (bias=False);
            ConvFormer and CAFormer utilizes LayerNorm without bias (bias=False).
    """

    def __init__(self, affine_shape=None, normalized_dim=(-1, ), 
                 scale=True, bias=True, eps=1e-5):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / (s + self.eps).sqrt()
        if self.use_scale:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x


class LayerNormWithoutBias(nn.Module):
    """
    Equal to partial(LayerNormGeneral, bias=False) but faster, 
    because it directly utilizes otpimized F.layer_norm
    """
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, None, self.eps)


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """
    def __init__(self, dim, expansion_ratio=2,
                 act1_layer=nn.ReLU, act2_layer=nn.Identity, 
                 bias=False, kernel_size=7, padding=3,
                 **kwargs, ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=padding, groups=med_channels, bias=bias) # depthwise conv
        self.act2 = act2_layer()
        self.pwconv2 = nn.Conv2d(med_channels, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act1(x)
        x = self.dwconv(x)
        x = self.act2(x)
        x = self.pwconv2(x)
        return x


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer: https://arxiv.org/abs/2111.11418
    Modfied for our use case.
    """
    def __init__(self, pool_size=3, **kwargs):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        y = self.pool(x)
        return y - x


class RandomMixing(nn.Module):
    def __init__(self, num_tokens=196, **kwargs):
        super().__init__()
        self.random_matrix = nn.parameter.Parameter(
            data=torch.softmax(torch.rand(num_tokens, num_tokens), dim=-1), 
            requires_grad=False)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1)
        x = torch.einsum('bij, mnj -> bimn', x, self.random_matrix)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False, **kwargs):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class MetaFormerBlock(nn.Module):
    """
    Implementation of one MetaFormer block.
    """
    def __init__(self, dim,
                 token_mixer=nn.Identity, mlp=SepConv,
                 norm_layer=LayerNormWithoutBias,
                 drop_path=0., layer_scale_init_value=None, 
                 res_scale_init_value=None,
                 **kwargs):
        super().__init__()

        # Use LayerNormGeneral for conv features instead of LayerNormWithoutBias
        if norm_layer == LayerNormWithoutBias:
            self.norm1 = LayerNormGeneral((dim, 1, 1), normalized_dim=(1,), eps=1e-6, bias=False)
            self.norm2 = LayerNormGeneral((dim, 1, 1), normalized_dim=(1,), eps=1e-6, bias=False)
        else:
            self.norm1 = norm_layer(dim)
            self.norm2 = norm_layer(dim)
            
        self.token_mixer = token_mixer(dim=dim, **kwargs)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = LayerScale(dim, layer_scale_init_value) \
            if layer_scale_init_value is not None else nn.Identity()
        self.res_scale1 = Scale(dim, res_scale_init_value) \
            if res_scale_init_value is not None else nn.Identity()

        self.mlp = mlp(dim=dim, **kwargs)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = LayerScale(dim, layer_scale_init_value) \
            if layer_scale_init_value is not None else nn.Identity()
        self.res_scale2 = Scale(dim, res_scale_init_value) \
            if res_scale_init_value is not None else nn.Identity()
        
    def forward(self, x):
        x = self.res_scale1(x) + \
            self.layer_scale1(
                self.drop_path1(
                    self.token_mixer(self.norm1(x))
                )
            )
        x = self.res_scale2(x) + \
            self.layer_scale2(
                self.drop_path2(
                    self.mlp(self.norm2(x))
                )
            )
        return x


class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """
    def __init__(self, dim, init_value=1.0, trainable=True, **kwargs):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), 
                                  requires_grad=trainable)

    def forward(self, x):
        # Handle both 2D (B, C) and 4D (B, C, H, W) tensors
        if x.dim() == 4:
            # Reshape scale for broadcasting: (C,) -> (1, C, 1, 1)
            scale = self.scale.view(1, -1, 1, 1)
        else:
            # For 2D tensors, use scale as is
            scale = self.scale
        return x * scale


def basic_blocks(dim, index, layers,
                 token_mixer=nn.Identity, mlp=SepConv,
                 norm_layer=LayerNormWithoutBias,
                 drop_path_rate=0., 
                 layer_scale_init_value=None, res_scale_init_value=None,
                 **kwargs):
    """
    generate PoolFormer blocks for a stage
    return: PoolFormer blocks 
    """
    blocks = []
    for block_idx in range(layers[index]):
        # Handle case where drop_path_rate is a list or single value
        if isinstance(drop_path_rate, (list, tuple)):
            block_dpr = drop_path_rate[block_idx] if block_idx < len(drop_path_rate) else 0.
        else:
            block_dpr = drop_path_rate * (
                block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(MetaFormerBlock(
            dim, token_mixer=token_mixer, mlp=mlp,
            norm_layer=norm_layer, drop_path=block_dpr, 
            layer_scale_init_value=layer_scale_init_value, 
            res_scale_init_value=res_scale_init_value,
            **kwargs
            ))
    blocks = nn.Sequential(*blocks)

    return blocks


class MetaFormer(nn.Module):
    r"""
    MetaFormer, the main class of our model
    --layers: [x,x,x,x], number of blocks for the 4 stages
    --embed_dims, --mlp_ratios, --act_layer: the embedding dims, mlp ratios and activation layer for each stage
    --num_classes: number of classes for the image classification
    --in_patch_size, --in_stride, --in_pad: specify the patch embedding
        for the input image
    --down_patch_size --down_stride --down_pad: 
        specify the downsample (patch embed.)
    --fork_faat: whether output features of the 4 stages, for dense prediction
    --init_cfg, --pretrained: 
        for mmdetection and mmsegmentation to load pretrianged weights
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[2, 2, 6, 2], dims=[64, 128, 320, 512],
                 downsample_layers=['stem', 'conv', 'conv', 'conv'],
                 token_mixers=nn.Identity, mlps=SepConv,
                 norm_layers=LayerNormWithoutBias, 
                 drop_path_rate=0.,
                 layer_scale_init_values=None,
                 res_scale_init_values=[None, None, 1.0, 1.0],
                 output_norm=partial(LayerNormGeneral, 
                                     normalized_dim=(1,), eps=1e-6, bias=False),
                 head_fn=nn.Linear,
                 **kwargs):
        super().__init__()
        if not isinstance(depths, (list, tuple)):
            depths = [depths] # it means the model has only one stage
        if not isinstance(dims, (list, tuple)):
            dims = [dims]

        num_stage = len(depths)
        self.num_stage = num_stage

        if not isinstance(downsample_layers, (list, tuple)):
            downsample_layers = [downsample_layers] * num_stage 
        down_dims = [in_chans] + dims
        self.downsample_layers = nn.ModuleList()
        for i in range(num_stage):
            if downsample_layers[i] == 'stem':
                layer = Stem(down_dims[i], down_dims[i+1])
            else:
                layer = Downsample(down_dims[i], down_dims[i+1])
            self.downsample_layers.append(layer)

        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * num_stage

        if not isinstance(mlps, (list, tuple)):
            mlps = [mlps] * num_stage

        if not isinstance(norm_layers, (list, tuple)):
            norm_layers = [norm_layers] * num_stage
            
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        if not isinstance(layer_scale_init_values, (list, tuple)):
            layer_scale_init_values = [layer_scale_init_values] * num_stage
        if not isinstance(res_scale_init_values, (list, tuple)):
            res_scale_init_values = [res_scale_init_values] * num_stage            

        self.stages = nn.ModuleList() # each stage consists of multiple metaformer blocks
        cur = 0
        for i in range(num_stage):
            stage = basic_blocks(dims[i], i, depths,
                                token_mixer=token_mixers[i], mlp=mlps[i],
                                norm_layer=norm_layers[i],
                                drop_path_rate=dp_rates[cur:cur + depths[i]], 
                                layer_scale_init_value=layer_scale_init_values[i],
                                res_scale_init_value=res_scale_init_values[i],
                            )
            self.stages.append(stage)
            cur += depths[i]

        self.fork_feat = True

        if self.fork_feat:
            self.out_indices = [0, 1, 2, 3]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    # TODO: more elegant way
                    """For RetinaNet, `start_level=1`. The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
                    layer = nn.Identity()
                else:
                    layer = output_norm((dims[i_emb], 1, 1))
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = output_norm((dims[-1], 1, 1))
            self.head = head_fn(dims[-1], num_classes) \
                if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_embeddings(self, x):
        embeddings = []
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if self.fork_feat:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                embeddings.append(x_out)
        if self.fork_feat:
            return embeddings

        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_embeddings(x)
        if self.fork_feat:
            # for dense prediction
            return x
        else:
            # for image classification
            x = self.head(x.mean([-2, -1]))
            return x


from functools import partial
import os


class Stem(nn.Module):
    """ 
    Stem implemented by a layer of convolution.
    Conv2d params constant across different model scales.
    """
    def __init__(self, in_dim, out_dim, act_layer=nn.GELU):
        super().__init__()        
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=7, stride=4, padding=2)
        self.norm = LayerNormGeneral((out_dim, 1, 1), normalized_dim=(1,), eps=1e-6, bias=False)
        self.act = act_layer()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Downsample(nn.Module):
    """ 
    Downsampling implemented by a layer of convolution.
    """
    def __init__(self, in_dim, out_dim,):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)
        self.norm = LayerNormGeneral((out_dim, 1, 1), normalized_dim=(1,), eps=1e-6, bias=False)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class MetaFormerUNet(nn.Module):
    def __init__(self, in_chans=3, out_chans=1, 
                 depths=[2, 2, 6, 2], dims=[64, 128, 320, 512],
                 token_mixers=[Pooling, Pooling, Pooling, Pooling], **kwargs):
        super().__init__()
        
        # Encoder
        self.encoder = MetaFormer(
            in_chans=in_chans,
            depths=depths,
            dims=dims,
            downsample_layers=['stem', 'conv', 'conv', 'conv'],
            token_mixers=token_mixers,
            mlps=SepConv,
            norm_layers=LayerNormWithoutBias,
            **kwargs
        )
        
        # Decoder
        self.decoder_layers = nn.ModuleList()
        for i in range(len(dims) - 1, 0, -1):
            decoder = nn.Sequential(
                nn.ConvTranspose2d(dims[i], dims[i-1], kernel_size=2, stride=2),
                nn.Conv2d(dims[i-1] * 2, dims[i-1], kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(dims[i-1], dims[i-1], kernel_size=3, padding=1),
                nn.GELU()
            )
            self.decoder_layers.append(decoder)
        
        # Final upsampling
        self.final_upsample = nn.ConvTranspose2d(dims[0], dims[0]//2, kernel_size=4, stride=4)
        self.final_conv = nn.Conv2d(dims[0]//2, out_chans, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        
        # Decoder with skip connections
        x = features[-1]
        for i, decoder in enumerate(self.decoder_layers):
            x = decoder[0](x)  # Upsample
            # Skip connection
            skip = features[-(i+2)]
            x = torch.cat([x, skip], dim=1)
            # Process concatenated features
            for j in range(1, len(decoder)):
                x = decoder[j](x)
        
        # Final layers
        x = self.final_upsample(x)
        x = self.final_conv(x)
        
        return x


def create_poolformerv2_s12(in_chans=3, out_chans=1, **kwargs):
    """Create PoolFormerV2-S12 UNet model"""
    model = MetaFormerUNet(
        in_chans=in_chans,
        out_chans=out_chans,
        depths=[2, 2, 6, 2],
        dims=[64, 128, 320, 512],
        token_mixers=[Pooling, Pooling, Pooling, Pooling],
        **kwargs
    )
    return model


def create_poolformerv2_s24(in_chans=3, out_chans=1, **kwargs):
    """Create PoolFormerV2-S24 UNet model"""
    model = MetaFormerUNet(
        in_chans=in_chans,
        out_chans=out_chans,
        depths=[4, 4, 12, 4],
        dims=[64, 128, 320, 512],
        token_mixers=[Pooling, Pooling, Pooling, Pooling],
        **kwargs
    )
    return model


def create_poolformerv2_s36(in_chans=3, out_chans=1, **kwargs):
    """Create PoolFormerV2-S36 UNet model"""
    model = MetaFormerUNet(
        in_chans=in_chans,
        out_chans=out_chans,
        depths=[6, 6, 18, 6],
        dims=[64, 128, 320, 512],
        token_mixers=[Pooling, Pooling, Pooling, Pooling],
        **kwargs
    )
    return model


def create_convformer_s18(in_chans=3, out_chans=1, **kwargs):
    """Create ConvFormer-S18 UNet model"""
    model = MetaFormerUNet(
        in_chans=in_chans,
        out_chans=out_chans,
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, SepConv, SepConv],
        **kwargs
    )
    return model


def create_convformer_s36(in_chans=3, out_chans=1, **kwargs):
    """Create ConvFormer-S36 UNet model"""
    model = MetaFormerUNet(
        in_chans=in_chans,
        out_chans=out_chans,
        depths=[3, 12, 18, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, SepConv, SepConv],
        **kwargs
    )
    return model


if __name__ == "__main__":
    # Test the model
    model = create_poolformerv2_s12(in_chans=3, out_chans=1)
    x = torch.randn(1, 3, 128, 128)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")