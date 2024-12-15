import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import torch.distributed as dist

from load_WMT import BASE_DIR, DATA_ROOT, LANGUAGE_PAIRS


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