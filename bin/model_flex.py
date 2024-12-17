'''
mod_model.py but with flex_attention and CasualAttention
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention
import math
from data_loader import MTConfig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FlexMultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, dim)
        self.c_v = CastedLinear(dim, dim)
        self.c_proj = CastedLinear(dim, dim)
        self.dropout = dropout

    def forward(self, x, mask=None, key_padding_mask=None):
        B, T = x.size(0), x.size(1)
        q = self.c_q(x).view(B, T, self.num_heads, -1)
        k = self.c_k(x).view(B, T, self.num_heads, -1)
        v = self.c_v(x).view(B, T, self.num_heads, -1)

        q, k = norm(q), norm(k)
        y = flex_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
            dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view_as(x)
        return self.c_proj(y)


class FlexCausalAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads

        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, dim)
        self.c_v = CastedLinear(dim, dim)
        self.c_proj = CastedLinear(dim, dim)
        self.dropout = dropout
        self.c_proj.weight.data.zero_()

    def forward(self, x, mask=None, key_padding_mask=None):
        B, T = x.size(0), x.size(1)

        q = self.c_q(x).view(B, T, self.num_heads, -1)
        k = self.c_k(x).view(B, T, self.num_heads, -1)
        v = self.c_v(x).view(B, T, self.num_heads, -1)

        q, k = norm(q), norm(k)

        y = flex_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0
        )
        
        y = y.transpose(1, 2).contiguous().view_as(x)
        return self.c_proj(y)


def norm(x):
    """RMS normalization from nanoGPT"""
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    """Memory-efficient linear layer that casts weights to input dtype"""
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)
        
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))


class RotaryEmbedding(nn.Module):
    """Rotary position embedding from nanoGPT"""
    def __init__(self, dim, base=10000):
        super().__init__()
        self.register_buffer('inv_freq', (1 / base) ** (torch.arange(0, dim, 2) / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            t = torch.arange(seq_len, device=x.device)
            freqs = torch.outer(t, self.inv_freq)
            self.seq_len_cached = seq_len
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        cos, sin = self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
        x1, x2 = x.chunk(2, dim=3)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x)


class FlexEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = FlexMultiHeadAttention(d_model, nhead, dropout)
        self.linear1 = CastedLinear(d_model, dim_feedforward)
        self.linear2 = CastedLinear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        x = x + self.dropout1(self.self_attn(self.norm1(x), src_mask, src_key_padding_mask))
        x = x + self.dropout2(self.linear2(F.relu(self.linear1(self.norm2(x)))))
        return x


class FlexDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = FlexCausalAttention(d_model, nhead, dropout)
        self.multihead_attn = FlexMultiHeadAttention(d_model, nhead, dropout)
        
        self.linear1 = CastedLinear(d_model, dim_feedforward)
        self.linear2 = CastedLinear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = tgt
        x = x + self.dropout1(self.self_attn(self.norm1(x), tgt_mask, tgt_key_padding_mask))
        x = x + self.dropout2(self.multihead_attn(self.norm2(x), memory_mask, memory_key_padding_mask))
        x = x + self.dropout3(self.linear2(F.relu(self.linear1(self.norm3(x)))))
        return x


class TransformerModel(nn.Module):
    def __init__(self, config: MTConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.d_model = config.model_dim
        self.nhead = config.num_heads
        self.dim_feedforward = config.dim_feedforward
        self.dropout = config.dropout
        self.num_encoder_layers = config.num_layers // 2
        self.num_decoder_layers = config.num_layers - self.num_encoder_layers
        
        self.src_embed = nn.Embedding(self.vocab_size, self.d_model)
        self.tgt_embed = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoder = RotaryEmbedding(self.d_model // self.nhead)
        
        # Use custom flex attention layers
        encoder_layer = FlexEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        )
        decoder_layer = FlexDecoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, self.num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, self.num_decoder_layers)
        
        self.output_layer = CastedLinear(self.d_model, self.vocab_size)
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.02)
        
        self._init_parameters()
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, 
                src_padding_mask=None, tgt_padding_mask=None):
        src_embedded = self.src_embed(src) * math.sqrt(self.d_model)
        tgt_embedded = self.tgt_embed(tgt) * math.sqrt(self.d_model)

        B, T, E = src_embedded.shape
        H = self.nhead
        src_reshaped = src_embedded.view(B, T, H, E//H)
        tgt_reshaped = tgt_embedded.view(B, T, H, E//H)
        
        src_encoded = self.pos_encoder(src_reshaped).view(B, T, E)
        tgt_encoded = self.pos_encoder(tgt_reshaped).view(B, T, E)

        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))
            tgt_mask = tgt_mask.to(tgt.device)

        src_encoded = norm(src_encoded)
        tgt_encoded = norm(tgt_encoded)

        memory = self.encoder(src_encoded, src_mask, src_padding_mask)

        output = self.decoder(
            tgt_encoded, memory,
            tgt_mask=tgt_mask,
            memory_mask=src_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )

        return self.output_layer(output)
    
    def _init_parameters(self):
        """Initialize parameters using Xavier uniform initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    @staticmethod
    def generate_square_subsequent_mask(sz):
        """Generate mask for decoder self-attention"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask

    def configure_optimizers(self, weight_decay, learning_rate, device_type, master_process=False, group_params=True):
        """Configure AdamW optimizer with optional parameter grouping and fused kernel usage."""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        if group_params:
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ]
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            if master_process:
                print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
                print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        else:
            optim_groups = [{"params": param_dict.values(), "weight_decay": weight_decay}]

        use_fused = hasattr(torch.optim.AdamW, 'fused') and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def check_only_model_params():
    config = MTConfig()
    model = TransformerModel(config)
    model.to(device)
    model = torch.compile(model)
    print("compiled model")
    print("model parameters:")
    tot = 0
    for name, param in model.named_parameters():
        tot += param.numel()
    print(f"total parameters: {tot / 1e6:.2f}M")


if __name__ == "__main__":
    check_only_model_params()
