import torch

def create_attention_mask(batch_size, num_heads, seq_length):
    # Create a mask of shape [batch_size, num_heads, seq_length, seq_length]
    mask = torch.ones(batch_size, num_heads, seq_length, seq_length)
    
    # Make it causal (lower triangular) for decoder self-attention
    mask = torch.triu(mask, diagonal=1).bool()
    mask = mask.masked_fill(mask == 1, float('-inf'))
    
    return mask

# Example usage
mask = create_attention_mask(batch_size=2, num_heads=4, seq_length=10)
print(mask.shape)