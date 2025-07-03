import torch
from performer_pytorch import PerformerLM
import torch.nn as nn


if __name__ == '__main__':
    model = PerformerLM(
        num_tokens=20000,
        max_seq_len=2048,             # max sequence length
        dim=512,                      # dimension
        depth=6,                      # layers
        heads=8,                      # heads
        causal=False,                 # auto-regressive or not
        # number of random features, if not set, will default to (d * log(d)), where d is the dimension of each head
        nb_features=256,
        # defaults to softmax approximation, but can be set to True for generalized attention
        generalized_attention=False,
        # the kernel function to be used, if generalized attention is turned on, defaults to Relu
        kernel_fn=nn.ReLU(),
        reversible=True,              # reversible layers, from Reformer paper
        ff_chunks=10,                 # chunk feedforward layer, from Reformer paper
        use_scalenorm=False,          # use scale norm, from 'Transformers without Tears' paper
        use_rezero=False,             # use rezero, from 'Rezero is all you need' paper
        # multiply final embeddings with token weights for logits, like gpt decoder
        # tie_embedding=False,
        ff_glu=True,                  # use GLU variant for feedforward
        emb_dropout=0.1,              # embedding dropout
        ff_dropout=0.1,               # feedforward dropout
        attn_dropout=0.1,             # post-attn dropout
    )
    x = torch.randint(0, 20000, (1, 2048))
    mask = torch.ones_like(x).bool()
    model(x, mask=mask)  # (1, 2048, 20000)
