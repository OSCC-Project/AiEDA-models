import torch
from torch import nn, einsum
from torch_geometric.nn import global_mean_pool
from .gnn_models import GNN
from torch_geometric.nn import MessagePassing
from einops import rearrange
from torch_geometric import utils
import torch_geometric.utils as utils
from torch_scatter import scatter_add, scatter_mean, scatter_max
import torch_geometric.nn as gnn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import math
from vit_pytorch.vit import Transformer


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# Norm layer (Norm)

class Norm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# Feedforward network (FFN)

class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# Attention-layer (MSA)

class MSA(nn.Module):
    def __init__(self, dim, heads = 5, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, kv_include_self = False):
        b, n, _, h = *x.shape, self.heads
        context = default(context, x)

        if kv_include_self:
            context = torch.cat((x, context), dim = 1) # CA needs the cls token to exchange information

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# Large and Small Patch Transformer encoder

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Norm(dim, MSA(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                Norm(dim, FFN(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

# Function to adjust and project small and large patch embedding tokens

class ProjectFunction(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn

        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x

# Cross attention transformer (CA)

class CATransformer(nn.Module):
    def __init__(self, sm_dim, lg_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ProjectFunction(sm_dim, lg_dim, Norm(lg_dim, MSA(lg_dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                ProjectFunction(lg_dim, sm_dim, Norm(sm_dim, MSA(sm_dim, heads = heads, dim_head = dim_head, dropout = dropout)))
            ]))

    def forward(self, sm_tokens, lg_tokens):
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (sm_tokens, lg_tokens))

        for sm_attend_lg, lg_attend_sm in self.layers:
            sm_cls = sm_attend_lg(sm_cls, context = lg_patch_tokens, kv_include_self = True) + sm_cls
            lg_cls = lg_attend_sm(lg_cls, context = sm_patch_tokens, kv_include_self = True) + lg_cls

        sm_tokens = torch.cat((sm_cls, sm_patch_tokens), dim = 1)
        lg_tokens = torch.cat((lg_cls, lg_patch_tokens), dim = 1)
        return sm_tokens, lg_tokens

# Multi-scale Transformer encoder

class MultiScaleTransformer(nn.Module):
    def __init__(
        self,
        *,
        depth,
        sm_dim,
        lg_dim,
        sm_enc_params,
        lg_enc_params,
        cross_attn_heads,
        cross_attn_depth,
        cross_attn_dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(dim = sm_dim, dropout = dropout, **sm_enc_params),
                Transformer(dim = lg_dim, dropout = dropout, **lg_enc_params),
                CATransformer(sm_dim = sm_dim, lg_dim = lg_dim, depth = cross_attn_depth, heads = cross_attn_heads, dim_head = cross_attn_dim_head, dropout = dropout)
            ]))

    def forward(self, sm_tokens, lg_tokens):
        for sm_enc, lg_enc, cross_attend in self.layers:
            sm_tokens, lg_tokens = sm_enc(sm_tokens), lg_enc(lg_tokens)
            sm_tokens, lg_tokens = cross_attend(sm_tokens, lg_tokens)

        return sm_tokens, lg_tokens

# Convert 1D image tokens into patch embeddings - Adaptation for 1D visual fatures

class Sequence1DEmbedder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        emb_size,
        patch_size,
        dropout = 0.
    ):
        super().__init__()
        
        emb_height, emb_width = pair(emb_size)
        patch_height, patch_width = pair(patch_size)

        num_patches = (emb_height // patch_height) * (emb_width // patch_width)
        patch_dim = 1 * patch_height * patch_width
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        return self.dropout(x)

# Cross-Attention Transformer adapted for 1D visual features - graph embeddings

class TR(nn.Module):
    def __init__(
        self,
        gnn,
        synth_encoder,
        batch_size,
        gnnemb_size,
        synemb_size,
        num_classes,
        sm_dim,
        lg_dim,
        sm_patch_size = (16,1),
        sm_enc_depth = 3,
        sm_enc_heads = 5,
        sm_enc_mlp_dim = 1024,
        sm_enc_dim_head = 64,
        lg_patch_size = (32,1),
        lg_enc_depth = 3,
        lg_enc_heads = 5,
        lg_enc_mlp_dim = 1024,
        lg_enc_dim_head = 64,
        cross_attn_depth = 2,
        cross_attn_heads = 5,
        cross_attn_dim_head = 64,
        depth = 3,
        dropout = 0.2,
        emb_dropout = 0.2,
    ):
        super().__init__()
        self.gnn = gnn
        self.synth_encoder = synth_encoder
        self.batch_size = batch_size
        self.gnnemb_size = gnnemb_size * 2 
        self.synemb_size = synemb_size
        self.hidden_dim = 256
        self.num_layers = 4
        self.dropout = dropout
        self.sm_image_embedder = Sequence1DEmbedder(dim = sm_dim, emb_size = self.gnnemb_size + self.synemb_size, patch_size = sm_patch_size, dropout = emb_dropout)
        self.lg_image_embedder = Sequence1DEmbedder(dim = lg_dim, emb_size = self.gnnemb_size + self.synemb_size, patch_size = lg_patch_size, dropout = emb_dropout)
        self.syn_sm_fc_layer = nn.Linear(self.synemb_size, sm_dim)
        self.syn_lg_fc_layer = nn.Linear(self.synemb_size, lg_dim)
        self.multi_scale_encoder = MultiScaleTransformer(
            depth = depth,
            sm_dim = sm_dim,
            lg_dim = lg_dim,
            cross_attn_heads = cross_attn_heads,
            cross_attn_dim_head = cross_attn_dim_head,
            cross_attn_depth = cross_attn_depth,
            sm_enc_params = dict(
                depth = sm_enc_depth,
                heads = sm_enc_heads,
                mlp_dim = sm_enc_mlp_dim,
                dim_head = sm_enc_dim_head
            ),
            lg_enc_params = dict(
                depth = lg_enc_depth,
                heads = lg_enc_heads,
                mlp_dim = lg_enc_mlp_dim,
                dim_head = lg_enc_dim_head
            ),
            dropout = dropout
        )

        self.sm_mlp_head = nn.Sequential(nn.LayerNorm(sm_dim), nn.Linear(sm_dim, 1))
        self.lg_mlp_head = nn.Sequential(nn.LayerNorm(lg_dim), nn.Linear(lg_dim, 1))

    def forward(self, batch):
        emb = self.gnn(batch)
        synthFlow = batch.opt_seq.reshape(-1, 20)
        synthesis_emb = self.synth_encoder(synthFlow)
        emb = emb.reshape(self.batch_size,1,self.gnnemb_size,1)

        small_tokens = self.sm_image_embedder(emb)
        large_tokens = self.lg_image_embedder(emb)

        syn_sm_emb_adjusted = self.syn_sm_fc_layer(synthesis_emb)
        syn_lg_emb_adjusted = self.syn_lg_fc_layer(synthesis_emb)

        combined_small_tokens = torch.cat((small_tokens, syn_sm_emb_adjusted), dim=1)
        combined_large_tokens = torch.cat((large_tokens, syn_lg_emb_adjusted), dim=1)

        small_tokens, large_tokens = self.multi_scale_encoder(combined_small_tokens, combined_large_tokens)
        sm_cls, lg_cls = map(lambda t: t[:, 0], (small_tokens, large_tokens))

        # 通过分类头
        sm_cls_logits = self.sm_mlp_head(sm_cls)
        lg_cls_logits = self.lg_mlp_head(lg_cls)

        return sm_cls_logits + lg_cls_logits

class SynthesisTransformerEncoder(nn.Module):
    def __init__(self, num_synthesis_actions, embedding_dim, num_layers, num_heads, ff_dim, dropout):
        super(SynthesisTransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(num_synthesis_actions, embedding_dim)
        encoder_layer = TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, synthesis_sequences):
        synthesis_embeddings = self.embedding(synthesis_sequences)
        synthesis_embeddings *= torch.sqrt(torch.tensor(self.embedding.embedding_dim, dtype=torch.float32))
        synthesis_embeddings = self.dropout(synthesis_embeddings)
        
        encoded_synthesis = self.transformer_encoder(synthesis_embeddings)
        
        return encoded_synthesis
    
if __name__ == "__main__":
    pass
