import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    def __init__(self, src_vocab_size, tgt_vocab_size, 
                 d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        # Embedding layers with more efficient linear projections
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        
        # Rotary position encoding instead of standard positional encoding
        self.pos_encoder = RotaryEmbedding(d_model // nhead)
        
        # Transformer with normalization and attention modifications
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-normalization like in nanoGPT
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Output projection with casting
        self.output_layer = CastedLinear(d_model, tgt_vocab_size)
        
        # Initialize parameters
        self._init_parameters()
        
        self.d_model = d_model
        self.nhead = nhead
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, 
                src_padding_mask=None, tgt_padding_mask=None):
        # Embed and scale input tokens
        src_embedded = self.src_embed(src) * math.sqrt(self.d_model)
        tgt_embedded = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        
        # Apply rotary position encoding
        # Reshape for rotary encoding
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
        
        # Apply RMS norm before transformer layers
        src_encoded = norm(src_encoded)
        tgt_encoded = norm(tgt_encoded)
        
        # Encoder
        memory = self.encoder(src_encoded, src_mask, src_padding_mask)
        
        # Decoder
        output = self.decoder(
            tgt_encoded, memory,
            tgt_mask=tgt_mask,
            memory_mask=src_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        
        # Project to vocabulary size with casting
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