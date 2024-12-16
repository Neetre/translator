'''
Imported optimizations from the nanoGPT model
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention
import math
from data_loader import MTConfig

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

class TransformerModel(nn.Module):
    def __init__(self, config: MTConfig):
        super(TransformerModel, self).__init__()
        self.vocab_size = config.vocab_size
        self.d_model = config.model_dim
        self.nhead = config.num_heads
        self.dim_feedforward = config.dim_feedforward
        self.dropout = config.dropout
        self.num_encoder_layers = config.num_layers // 2
        self.num_decoder_layers = config.num_layers - self.num_encoder_layers
        
        # Embedding layers with more efficient linear projections
        self.src_embed = nn.Embedding(self.vocab_size, self.d_model)
        self.tgt_embed = nn.Embedding(self.vocab_size, self.d_model)
        
        # Rotary position encoding instead of standard positional encoding
        self.pos_encoder = RotaryEmbedding(self.d_model // self.nhead)
        
        # Transformer with normalization and attention modifications
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True,
            norm_first=True  # Pre-normalization like in nanoGPT
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, self.num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, self.num_decoder_layers)

        # Output projection with casting
        self.output_layer = CastedLinear(self.d_model, self.vocab_size)
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.02)

        self._init_parameters()
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, 
                src_padding_mask=None, tgt_padding_mask=None):
        src_embedded = self.src_embed(src) * math.sqrt(self.d_model)
        tgt_embedded = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        
        # Apply rotary position encoding
        B, T, E = src_embedded.shape
        H = self.nhead
        src_reshaped = src_embedded.view(B, T, H, E//H)
        tgt_reshaped = tgt_embedded.view(B, T, H, E//H)
        
        src_encoded = self.pos_encoder(src_reshaped).view(B, T, E)
        tgt_encoded = self.pos_encoder(tgt_reshaped).view(B, T, E)
        
        # Generate masks if not provided
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
