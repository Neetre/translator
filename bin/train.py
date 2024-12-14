import os
from dataclasses import dataclass
from torch.utils.data import DataLoader
import torch
import numpy as np

from load_WMT import BASE_DIR, DATA_ROOT, LANGUAGE_PAIRS
from mod_model import TransformerModel


def get_data_dir(lang_pair, split):
    return os.path.join(DATA_ROOT, lang_pair, split)


def load_tokens(filename):
    npt = np.load(filename)
    ppt = torch.tensor(npt, dtype=torch.long)
    return ppt

class DataLoader:
    def __init__(self, B, T, process_rank, num_process, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_process = num_process
        assert split in {'train', 'val'}

        self.split = split
        self.shards = []
        for lang_pair in LANGUAGE_PAIRS:
            data_dir = get_data_dir(lang_pair, split)
            shard_files = sorted(os.listdir(data_dir))
            for shard_file in shard_files:
                shard = load_tokens(os.path.join(data_dir, shard_file))
                self.shards.append(shard)
        self.shards = sorted(self.shards)
        assert len(self.shards) > 0 , f"No shards found for {split}"
        if master_process:
            print(f"Found {len(self.shards)} shards for {split}")

        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):








@dataclass
class MTConfig:
    vocab_size : int = 100352 # 100257 is not divisible by 128 so better 100352 tiktoken = cl100k_base
    num_layers : int = 12  # 6 encoder + 6 decoder
    num_heads : int = 6 # head dim 128
    model_dim : int = 1024
    dim_feedforward : int = 4096
    dropout : float = 0.1

'''
src_vocab_size, tgt_vocab_size, 
                 d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1
'''

@dataclass
class Hyper:
    pass