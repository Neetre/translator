import numpy as np
import torch
import tiktoken
from transformers import T5Tokenizer
from transformers import MT5Tokenizer


src_path = '../data/opus/train/src_shard_000001.npy'
tgt_path = '../data/opus/train/trg_shard_000001.npy'


def get_indicies(src_path, tgt_path, EOT_ID=1):
    current_src = torch.from_numpy(np.load(src_path)).long()
    current_tgt = torch.from_numpy(np.load(tgt_path)).long()
    src_eot_indices = (current_src == EOT_ID).nonzero().squeeze(-1)
    tgt_eot_indices = (current_tgt == EOT_ID).nonzero().squeeze(-1)

    src_eot_indices = torch.cat([src_eot_indices, torch.tensor([len(current_src)])])
    tgt_eot_indices = torch.cat([tgt_eot_indices, torch.tensor([len(current_tgt)])])

    return current_src, current_tgt, src_eot_indices, tgt_eot_indices


def debug_prints(data, EOT_ID=1):
    current_src, current_tgt, src_eot_indices, tgt_eot_indices = data
    print(f"Source length: {len(current_src)} tokens")
    print(f"Target length: {len(current_tgt)} tokens")
    print(f"Source EOT indices: {src_eot_indices}")
    print(f"Target EOT indices: {tgt_eot_indices}")
    print(f"Source EOT count: {(current_src == EOT_ID).sum().item()}")
    print(f"Target EOT count: {(current_tgt == EOT_ID).sum().item()}")

    assert len(src_eot_indices) == len(tgt_eot_indices), "Mismatched number of source and target sequences"

def get_first_tokens(src_shard, tgt_shard):
    print(f"Source shard (first 100 tokens): {src_shard[:100]}")
    print(f"Target shard (first 100 tokens): {tgt_shard[:100]}")


def test_idx(current_src, current_tgt, src_eot_indices, tgt_eot_indices):
    idx = 1
    src_start = 0 if idx == 0 else src_eot_indices[idx - 1] + 1
    src_end = src_eot_indices[idx]
    tgt_start = 0 if idx == 0 else tgt_eot_indices[idx - 1] + 1
    tgt_end = tgt_eot_indices[idx]

    src = current_src[src_start:src_end]
    tgt = current_tgt[tgt_start:tgt_end]

    print(f"Source sequence: {src}")
    print(f"Target sequence: {tgt}")
    print(current_src[src_start])


def initialize_tokenizer(model_name="t5-base"):
    """Initialize the T5 tokenizer and get EOT token ID."""
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    eot_id = tokenizer.eos_token_id
    print(f"End of text token ID: {eot_id}")
    return tokenizer, eot_id

def load_shards(src_path, tgt_path):
    """Load source and target shards from numpy files."""
    src_shard = np.load(src_path)
    tgt_shard = np.load(tgt_path)
    return src_shard, tgt_shard



def test_MT5Tokenizer():
    enc = MT5Tokenizer.from_pretrained("google/mt5-base")
    eot = enc.eos_token_id
    print(f"End of text token ID: {eot}")



def main():
    test_MT5Tokenizer()
    # Initialize tokenizer
    tokenizer, eot_id = initialize_tokenizer()
    
    # Define paths
    src_path = '../data/opus/train/src_shard_000001.npy'
    tgt_path = '../data/opus/train/trg_shard_000001.npy'

    # Load data
    src_shard, tgt_shard = load_shards(src_path, tgt_path)
    get_first_tokens(src_shard, tgt_shard)

    # Get EOT indices
    data = get_indicies(src_path, tgt_path)  # current_src, current_tgt, src_eot_indices, tgt_eot_indices
    test_idx(*data)
    

if __name__ == "__main__":
    main()