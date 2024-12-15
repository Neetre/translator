import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import torch.distributed as dist

from load_WMT import BASE_DIR, DATA_ROOT, LANGUAGE_PAIRS
from mod_model import TransformerModel


@dataclass
class MTConfig:
    vocab_size : int = 100352 # 100257 is not divisible by 128 so better 100352 tiktoken = cl100k_base
    num_layers : int = 12  # 6 encoder + 6 decoder
    num_heads : int = 6 # head dim 128
    model_dim : int = 1024
    dim_feedforward : int = 4096
    dropout : float = 0.1
    pad_token: int = 0
    max_seq_len: int = 1024


def get_data_dir(lang_pair, split):
    return os.path.join(DATA_ROOT, lang_pair, split)


def load_tokens(filename):
    npt = np.load(filename)
    ppt = torch.tensor(npt, dtype=torch.long)
    return ppt

class DataLoader:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
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

        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B , T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y



class MTDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 split: str,
                 max_seq_len: int,
                 rank: Optional[int] = None,
                 world_size: Optional[int] = None,
                 ):
        self.data_dir = data_dir
        self.split = split
        self.max_seq_len = max_seq_len
        self.rank = rank
        self.world_size = world_size

        self.src_shards = sorted([f for f in os.listdir(data_dir) if f.startswith('src_')])
        self.tgt_shards = sorted([f for f in os.listdir(data_dir) if f.startswith('tgt_')])
        assert len(self.src_shards) == len(self.tgt_shards), "Mismatched number of source and target shards"

        if rank is not None and world_size is not None:
            self.src_shards = self.src_shards[rank::world_size]
            self.tgt_shards = self.tgt_shards[rank::world_size]

        self.current_shard_idx = 0
        self.load_shard(0)

        if rank in {0, None}:
            print(f"Loaded dataset for {split} split with {len(self.src_shards)} shards")

    def load_shard(self, shard_idx: int) -> None:
        pass

    def __len__(self):
        return self.current_size
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        pass



@dataclass
class Hyper:
    pass