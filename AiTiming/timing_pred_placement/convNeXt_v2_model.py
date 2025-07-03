import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
import math


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    """ ConvNeXt V2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXtV2Encoder(nn.Module):
    def __init__(self, in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        
        # Stem layer
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        
        # 3 intermediate downsampling conv layers
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtV2Block(dim=dims[i], drop_path=dp_rates[cur + j]) 
                for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = LayerNorm(dims[-1], eps=1e-6, data_format="channels_first") # final norm layer
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            features.append(x)
        
        return features


class ConvNeXtV2UNet(nn.Module):
    def __init__(self, in_chans=3, out_chans=1, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
        super().__init__()
        self.encoder = ConvNeXtV2Encoder(in_chans, depths, dims)
        
        # Decoder
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        # Build decoder layers
        for i in range(3, 0, -1):
            # Upsample layer
            upsample = nn.ConvTranspose2d(dims[i], dims[i-1], kernel_size=2, stride=2)
            self.upsamples.append(upsample)
            
            # Decoder block
            decoder = nn.Sequential(
                nn.Conv2d(dims[i-1] * 2, dims[i-1], kernel_size=3, padding=1),
                LayerNorm(dims[i-1], eps=1e-6, data_format="channels_first"),
                nn.GELU(),
                ConvNeXtV2Block(dims[i-1])
            )
            self.decoders.append(decoder)
        
        # Final layers
        self.final_upsample = nn.ConvTranspose2d(dims[0], dims[0]//2, kernel_size=4, stride=4)
        self.final_conv = nn.Sequential(
            nn.Conv2d(dims[0]//2, dims[0]//4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dims[0]//4, out_chans, kernel_size=1)
        )
        
    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        
        # Decoder
        x = features[-1]
        for i, (upsample, decoder) in enumerate(zip(self.upsamples, self.decoders)):
            x = upsample(x)
            # Skip connection
            skip = features[-(i+2)]
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        
        # Final layers
        x = self.final_upsample(x)
        x = self.final_conv(x)
        
        return x


def create_convnextv2_tiny(in_chans=3, out_chans=1, **kwargs):
    """Create ConvNeXt V2 Tiny model"""
    model = ConvNeXtV2UNet(
        in_chans=in_chans, 
        out_chans=out_chans,
        depths=[3, 3, 9, 3], 
        dims=[96, 192, 384, 768],
        **kwargs
    )
    return model


def create_convnextv2_small(in_chans=3, out_chans=1, **kwargs):
    """Create ConvNeXt V2 Small model"""
    model = ConvNeXtV2UNet(
        in_chans=in_chans, 
        out_chans=out_chans,
        depths=[3, 3, 27, 3], 
        dims=[96, 192, 384, 768],
        **kwargs
    )
    return model


def create_convnextv2_base(in_chans=3, out_chans=1, **kwargs):
    """Create ConvNeXt V2 Base model"""
    model = ConvNeXtV2UNet(
        in_chans=in_chans, 
        out_chans=out_chans,
        depths=[3, 3, 27, 3], 
        dims=[128, 256, 512, 1024],
        **kwargs
    )
    return model


if __name__ == "__main__":
    # Test the model
    model = create_convnextv2_tiny(in_chans=3, out_chans=1)
    x = torch.randn(1, 3, 128, 128)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")