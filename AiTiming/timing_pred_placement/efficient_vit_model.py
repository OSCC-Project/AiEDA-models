import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * c.groups, w.size(
            0), w.shape[2:], stride=c.stride, padding=c.padding, dilation=c.dilation, groups=c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        torch.nn.init.trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - bn.running_mean * \
            bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class PatchMerging(torch.nn.Module):
    def __init__(self, dim, out_dim, input_resolution):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0)
        self.act = torch.nn.ReLU()
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim)
        self.se = SqueezeExcite(hid_dim, 0.25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x


class CascadedGroupAttention(torch.nn.Module):
    r""" Cascaded Group Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14,
                 kernels=[5, 5, 5, 5],):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio

        # Ensure kernels list has enough elements
        if len(kernels) < num_heads:
            kernels = kernels + [kernels[-1]] * (num_heads - len(kernels))

        qkvs = []
        dws = []
        for i in range(num_heads):
            qkvs.append(Conv2d_BN(dim // (num_heads), self.key_dim * 2 + self.d))
            dws.append(Conv2d_BN(self.key_dim, self.key_dim, kernels[i], 1, kernels[i]//2, groups=self.key_dim))
        self.qkvs = torch.nn.ModuleList(qkvs)
        self.dws = torch.nn.ModuleList(dws)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            self.d * num_heads, dim, bn_weight_init=0))

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,C,H,W)
        B, C, H, W = x.shape
        trainingab = self.attention_biases[:, self.attention_bias_idxs]
        feats_in = x.chunk(len(self.qkvs), dim=1)
        feats_out = []
        feat = feats_in[0]
        for i, qkv in enumerate(self.qkvs):
            if i > 0: # add the previous output to the input
                feat = feat + feats_in[i]
            feat = qkv(feat)
            q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1) # B, C/h, H, W
            q = self.dws[i](q)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2) # B, C/h, N
            attn = (
                (q.transpose(-2, -1) @ k) * self.scale
                +
                (trainingab[i] if self.training else self.ab[i])
            )
            attn = attn.softmax(dim=-1) # BNN
            feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W) # BCHW
            feats_out.append(feat)
        x = self.proj(torch.cat(feats_out, 1))
        return x


import itertools


class LocalWindowAttention(torch.nn.Module):
    r""" Local Window Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """

    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5],):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.resolution = resolution
        self.window_resolution = window_resolution
        self.attn_ratio = attn_ratio
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.scale = key_dim ** -0.5

        if type(kernels) == list:
            assert len(kernels) == num_heads
            self.kernels = kernels
        else:
            kernels = [kernels] * num_heads
            self.kernels = kernels

        self.qkv = Conv2d_BN(dim, self.num_heads * (self.key_dim + self.key_dim + self.d))
        
        self.N = resolution
        self.N2 = self.N ** 2
        self.Wh, self.Ww = window_resolution, window_resolution
        self.Ws = self.Wh * self.Ww # number of windows
        
        self.dws = torch.nn.ModuleList([
            Conv2d_BN(self.key_dim, self.key_dim, ks=kernel, stride=1, pad=kernel//2, groups=self.key_dim)
            for kernel in kernels
        ])
        
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            self.d * num_heads, dim, bn_weight_init=0))

        # positional encoding
        SEQ_L = (resolution // window_resolution) ** 2
        self.pos_emb = nn.Parameter(torch.zeros(num_heads, self.Ws, self.Ws))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Pad if necessary to make H and W divisible by window size
        pad_h = (self.Wh - H % self.Wh) % self.Wh
        pad_w = (self.Ww - W % self.Ww) % self.Ww
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H, W = H + pad_h, W + pad_w
        
        qkv = self.qkv(x)
        qkv = qkv.view(B, self.num_heads, self.key_dim + self.key_dim + self.d, H, W)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.d], dim=2) # B, num_heads, C, H, W
            
        # apply dwconv on queries - create new tensor to avoid in-place modification
        q_processed = []
        for i in range(self.num_heads):
            q_processed.append(self.dws[i](q[:, i]))
        q = torch.stack(q_processed, dim=1)
        
        # window partition, BHCWH -> B H//Wh W//Ww Wh*Ww C
        q = q.reshape(B, self.num_heads, self.key_dim, H//self.Wh, self.Wh, W//self.Ww, self.Ww).permute(0, 1, 3, 5, 2, 4, 6)
        k = k.reshape(B, self.num_heads, self.key_dim, H//self.Wh, self.Wh, W//self.Ww, self.Ww).permute(0, 1, 3, 5, 2, 4, 6)
        v = v.reshape(B, self.num_heads, self.d, H//self.Wh, self.Wh, W//self.Ww, self.Ww).permute(0, 1, 3, 5, 2, 4, 6)
        
        # local window attention
        q = q.flatten(-2, -1)  # B, num_heads, H//Wh, W//Ww, key_dim, Wh*Ww
        k = k.flatten(-2, -1)  # B, num_heads, H//Wh, W//Ww, key_dim, Wh*Ww
        v = v.flatten(-2, -1)  # B, num_heads, H//Wh, W//Ww, d, Wh*Ww
        
        # Transpose to get correct dimensions for attention
        q = q.transpose(-2, -1)  # B, num_heads, H//Wh, W//Ww, Wh*Ww, key_dim
        k = k.transpose(-2, -1)  # B, num_heads, H//Wh, W//Ww, Wh*Ww, key_dim
        v = v.transpose(-2, -1)  # B, num_heads, H//Wh, W//Ww, Wh*Ww, d
        
        attn = (q @ k.transpose(-2, -1)) * self.scale + self.pos_emb[:, None, None, :, :]
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).reshape(B, self.num_heads, H//self.Wh, W//self.Ww, self.Wh, self.Ww, self.d)
        x = x.permute(0, 1, 6, 2, 4, 3, 5).reshape(B, self.num_heads * self.d, H, W)
        
        # Remove padding if it was added
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H-pad_h, :W-pad_w]
        
        x = self.proj(x)
        return x


class EfficientViTBlock(torch.nn.Module):
    """ A EfficientViT Block.

    Args:
        dim (int): Number of input channels.
        key_dim (int): Dimension for query and key in the token mixer.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
    """

    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5]):
        super().__init__()

        self.dw0 = Residual(Conv2d_BN(dim, dim, 3, 1, 1, groups=dim, bn_weight_init=0.))
        self.ffn0 = Residual(FFN(dim, int(dim * 2)))

        if window_resolution == 0 or window_resolution >= resolution:
            self.mixer = Residual(CascadedGroupAttention(dim, key_dim, num_heads,
                                attn_ratio=attn_ratio,
                                resolution=resolution,
                                kernels=kernels))

        else:
            self.mixer = Residual(LocalWindowAttention(dim, key_dim, num_heads,
                            attn_ratio=attn_ratio,
                            resolution=resolution,
                            window_resolution=window_resolution,
                            kernels=kernels))

        self.dw1 = Residual(Conv2d_BN(dim, dim, 3, 1, 1, groups=dim, bn_weight_init=0.))
        self.ffn1 = Residual(FFN(dim, int(dim * 2)))

    def forward(self, x):
        return self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))


class EfficientViT(torch.nn.Module):
    def __init__(self, img_size=224,
                 patch_size=16,
                 num_classes=1000,
                 stages=['s', 's', 's'],
                 embed_dim=[64, 128, 192],
                 key_dim=[16, 16, 16],
                 depth=[1, 2, 3],
                 num_heads=[4, 4, 4],
                 window_size=[7, 7, 7],
                 kernels=[5, 5, 5, 5],
                 down_ops=[['subsample', 2], ['subsample', 2], ['']],
                 distillation=True,
                 ):
        super().__init__()

        resolution = img_size
        self.patch_embed = torch.nn.Sequential(Conv2d_BN(3, embed_dim[0] // 8, 3, 2, 1), torch.nn.ReLU(),
                           Conv2d_BN(embed_dim[0] // 8, embed_dim[0] // 4, 3, 2, 1), torch.nn.ReLU(),
                           Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1), torch.nn.ReLU(),
                           Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 1, 1))
        resolution = img_size // patch_size
        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]
        self.blocks1 = []
        self.blocks2 = []
        self.blocks3 = []
        for i, (stg, ed, kd, dpth, nh, ar, wd, do) in enumerate(
                zip(stages, embed_dim, key_dim, depth, num_heads, attn_ratio, window_size, down_ops)):
            for d in range(dpth):
                eval('self.blocks' + str(i + 1)).append(EfficientViTBlock(ed, kd, nh, ar, resolution, wd, kernels))
            if do[0] == 'subsample':
                #('Subsample' stride)
                blk = PatchMerging(embed_dim[i], embed_dim[i + 1] if i < len(embed_dim) - 1 else embed_dim[i], resolution)
                resolution = resolution // do[1]
                eval('self.blocks' + str(i + 1)).append(blk)

        self.blocks1 = torch.nn.Sequential(*self.blocks1)
        self.blocks2 = torch.nn.Sequential(*self.blocks2)
        self.blocks3 = torch.nn.Sequential(*self.blocks3)

    def forward(self, x):
        x = self.patch_embed(x)
        x1 = self.blocks1(x)
        x2 = self.blocks2(x1)
        x3 = self.blocks3(x2)
        return [x, x1, x2, x3]


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=torch.sigmoid, divisor=1, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class EfficientViTUNet(nn.Module):
    def __init__(self, in_chans=3, out_chans=1, img_size=128, **kwargs):
        super().__init__()
        
        # Encoder (EfficientViT backbone)
        self.encoder = EfficientViT(
            img_size=img_size,
            patch_size=8,  # Smaller patch size for better resolution
            stages=['s', 's', 's'],
            embed_dim=[64, 128, 256],
            key_dim=[16, 16, 16],
            depth=[2, 2, 4],
            num_heads=[4, 4, 8],
            window_size=[7, 7, 7],
            down_ops=[['subsample', 2], ['subsample', 2], ['']],
        )
        
        # Decoder
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1),  # After concatenation: 256 + 128 = 384
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.Conv2d(64 + 128, 64, kernel_size=3, padding=1),  # After concatenation: 64 + 128 = 192
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.Conv2d(32 + 64, 32, kernel_size=3, padding=1),  # After concatenation: 32 + 64 = 96
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Final layer
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=8, stride=8),  # Match patch_size
            nn.Conv2d(16, out_chans, kernel_size=1)
        )
        
    def forward(self, x):
        # Encoder
        features = self.encoder(x)  # [x, x1, x2, x3]
        
        # Decoder with skip connections
        d3 = self.decoder3[0](features[3])  # Upsample
        # Ensure spatial dimensions match for concatenation
        if d3.shape[2:] != features[2].shape[2:]:
            d3 = F.interpolate(d3, size=features[2].shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, features[2]], dim=1)  # Skip connection
        for layer in self.decoder3[1:]:
            d3 = layer(d3)
        
        d2 = self.decoder2[0](d3)  # Upsample
        # Ensure spatial dimensions match for concatenation
        if d2.shape[2:] != features[1].shape[2:]:
            d2 = F.interpolate(d2, size=features[1].shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, features[1]], dim=1)  # Skip connection
        for layer in self.decoder2[1:]:
            d2 = layer(d2)
        
        d1 = self.decoder1[0](d2)  # Upsample
        # Ensure spatial dimensions match for concatenation
        if d1.shape[2:] != features[0].shape[2:]:
            d1 = F.interpolate(d1, size=features[0].shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, features[0]], dim=1)  # Skip connection
        for layer in self.decoder1[1:]:
            d1 = layer(d1)
        
        # Final output
        out = self.final_layer(d1)
        
        return out


def create_efficientvit_m0(in_chans=3, out_chans=1, img_size=128, **kwargs):
    """Create EfficientViT-M0 UNet model"""
    model = EfficientViTUNet(in_chans=in_chans, out_chans=out_chans, img_size=img_size, **kwargs)
    return model


def create_efficientvit_m1(in_chans=3, out_chans=1, img_size=128, **kwargs):
    """Create EfficientViT-M1 UNet model"""
    class EfficientViTM1UNet(EfficientViTUNet):
        def __init__(self, in_chans=3, out_chans=1, img_size=128, **kwargs):
            super(EfficientViTUNet, self).__init__()
            
            self.encoder = EfficientViT(
                img_size=img_size,
                patch_size=8,
                stages=['s', 's', 's'],
                embed_dim=[128, 144, 192],
                key_dim=[16, 16, 16],
                depth=[2, 3, 3],
                num_heads=[4, 4, 6],
                window_size=[7, 7, 7],
                down_ops=[['subsample', 2], ['subsample', 2], ['']],
            )
            
            # Update decoder for new dimensions
            # embed_dim=[128, 144, 192] for M1
            # features: [x(128ch), x1(128ch), x2(144ch), x3(192ch)]
            self.decoder3 = nn.Sequential(
                nn.ConvTranspose2d(192, 144, kernel_size=2, stride=2),
                nn.Conv2d(192 + 144, 144, kernel_size=3, padding=1),  # After concatenation: 192 + 144 = 336
                nn.ReLU(inplace=True),
                nn.Conv2d(144, 144, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            
            self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d(144, 128, kernel_size=2, stride=2),
                nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1),  # After concatenation: 128 + 128 = 256
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            
            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                nn.Conv2d(64 + 128, 64, kernel_size=3, padding=1),  # After concatenation: 64 + 128 = 192
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            
            self.final_layer = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=8, stride=8),
                nn.Conv2d(32, out_chans, kernel_size=1)
            )
    
    return EfficientViTM1UNet(in_chans=in_chans, out_chans=out_chans, img_size=img_size, **kwargs)


def create_efficientvit_m2(in_chans=3, out_chans=1, img_size=128, **kwargs):
    """Create EfficientViT-M2 UNet model"""
    class EfficientViTM2UNet(EfficientViTUNet):
        def __init__(self, in_chans=3, out_chans=1, img_size=128, **kwargs):
            super(EfficientViTUNet, self).__init__()
            
            self.encoder = EfficientViT(
                img_size=img_size,
                patch_size=8,
                stages=['s', 's', 's'],
                embed_dim=[128, 192, 224],
                key_dim=[16, 16, 16],
                depth=[2, 3, 5],
                num_heads=[4, 6, 7],
                window_size=[7, 7, 7],
                down_ops=[['subsample', 2], ['subsample', 2], ['']],
            )
            
            # Update decoder for new dimensions
            self.decoder3 = nn.Sequential(
                nn.ConvTranspose2d(224, 192, kernel_size=2, stride=2),
                nn.Conv2d(384, 192, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 192, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            
            self.decoder2 = nn.Sequential(
                nn.ConvTranspose2d(192, 128, kernel_size=2, stride=2),
                nn.Conv2d(320, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            
            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                nn.Conv2d(192, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            
            self.final_layer = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=8, stride=8),
                nn.Conv2d(32, out_chans, kernel_size=1)
            )
    
    return EfficientViTM2UNet(in_chans=in_chans, out_chans=out_chans, img_size=img_size, **kwargs)


if __name__ == "__main__":
    # Test the model
    model = create_efficientvit_m0(in_chans=3, out_chans=1, img_size=128)
    x = torch.randn(1, 3, 128, 128)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")