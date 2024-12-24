import numpy as np
import torch
import tiktoken
from transformers import T5Tokenizer
model_name = "t5-base"
enc = T5Tokenizer.from_pretrained(model_name)
EOT_ID = enc.eos_token_id
# print(enc.decode(EOT_ID))
# enc = tiktoken.get_encoding("cl100k_base")
# EOT_ID = enc._special_tokens["<|endoftext|>"] 
print(f"End of text token ID: {EOT_ID}")

# Example paths to the shard files
src_path = '../data/opus/train/src_shard_000001.npy'
tgt_path = '../data/opus/train/trg_shard_000001.npy'

# Load the shards
src_shard = np.load(src_path)
tgt_shard = np.load(tgt_path)


# Print the first few tokens of each shard
print(f"Source shard (first 100 tokens): {src_shard[:100]}")
print(f"Target shard (first 100 tokens): {tgt_shard[:100]}")

# Print the last few tokens of each shard
# print(f"Source shard (last 100 tokens): {src_shard[-100:]}")
# print(f"Target shard (last 100 tokens): {tgt_shard[-100:]}")

'''
# Check for padding or extra tokens
src_unique_tokens = np.unique(src_shard)
tgt_unique_tokens = np.unique(tgt_shard)

print(f"Unique source tokens: {src_unique_tokens}")
print(f"Unique target tokens: {tgt_unique_tokens}")
'''

current_src = torch.from_numpy(np.load(src_path)).long()
current_tgt = torch.from_numpy(np.load(tgt_path)).long()
src_eot_indices = (current_src == EOT_ID).nonzero().squeeze(-1)
tgt_eot_indices = (current_tgt == EOT_ID).nonzero().squeeze(-1)
# Add the end index to make slicing easier
src_eot_indices = torch.cat([src_eot_indices, torch.tensor([len(current_src)])])
tgt_eot_indices = torch.cat([tgt_eot_indices, torch.tensor([len(current_tgt)])])
# Debugging statements
print(f"Source length: {len(current_src)} tokens")
print(f"Target length: {len(current_tgt)} tokens")
print(f"Source EOT indices: {src_eot_indices}")
print(f"Target EOT indices: {tgt_eot_indices}")
print(f"Source EOT count: {(current_src == EOT_ID).sum().item()}")
print(f"Target EOT count: {(current_tgt == EOT_ID).sum().item()}")
assert len(src_eot_indices) == len(tgt_eot_indices), "Mismatched number of source and target sequences"

src_tokens = enc(
        "Hello, how are you?",
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )["input_ids"].squeeze(0)

src_tokens_numpy = src_tokens.numpy()
src_tokens_tolist = [EOT_ID] + src_tokens.numpy().tolist()

print(f"Source tokens (tensor): {src_tokens}")
print(f"Source tokens (list): {src_tokens_tolist}")
print(f"Source tokens (numpy): {src_tokens_numpy}")
print(f"Decoded source tokens: {enc.decode(src_tokens_tolist)}")  # Decoded source tokens: </s> Hello, how are you?</s>
print(f"Decoded source tokens (numpy): {enc.decode(src_tokens_numpy)}")  # Decoded source tokens (numpy): Hello, how are you?</s>


