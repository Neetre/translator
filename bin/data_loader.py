import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.distributed import init_process_group
from typing import List, Dict, Optional
from dataclasses import dataclass

from load_WMT import DATA_ROOT


def init_distributed():
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")

    device_type = "cuda" if device.startswith("cuda") else "cpu"
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process, device_type

ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process, device_type = init_distributed()

@dataclass
class MTConfig:
    vocab_size : int = 100352 # 100257 is not divisible by 128 so better 100352 tiktoken = cl100k_base
    num_layers : int = 8  # 4 encoder + 4 decoder
    num_heads : int = 4 # head dim 128
    model_dim : int = 512
    dim_feedforward : int = 1024
    dropout : float = 0.1
    pad_token: int = 0
    max_seq_len: int = 256  # 512 for now, but will be 1024
    batch_size: int = 64
    use_compiler: bool = False


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
        src_path = os.path.join(self.data_dir, self.src_shards[shard_idx])
        tgt_path = os.path.join(self.data_dir, self.tgt_shards[shard_idx])
        self.current_src = torch.from_numpy(np.load(src_path)).long()
        self.current_tgt = torch.from_numpy(np.load(tgt_path)).long()
        assert len(self.current_src) == len(self.current_tgt), "Mismatched number of source and target tokens"
        self.current_size = len(self.current_src)

    def __len__(self):
        return self.current_size
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if idx >= self.current_size:
            self.current_shard_idx = (self.current_shard_idx + 1) % len(self.src_shards)
            self.load_shard(self.current_shard_idx)
            idx = idx % self.current_size

        src = self.current_src[idx]
        tgt = self.current_tgt[idx]

        return {
            "source": src[:self.max_seq_len],
            "target": tgt[:self.max_seq_len]
        }
    
def create_padding_mask(batch: torch.Tensor, pad_token: int) -> torch.Tensor:
    return (batch == pad_token).unsqueeze(1).unsqueeze(2)


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token: int = 0) -> Dict[str, torch.Tensor]:
    max_src_len = max(len(item['source']) for item in batch)
    max_tgt_len = max(len(item['target']) for item in batch)

    batch_size = len(batch)
    src_padded = torch.full((batch_size, max_src_len), pad_token, dtype=torch.long)
    tgt_padded = torch.full((batch_size, max_tgt_len), pad_token, dtype=torch.long)

    for i, item in enumerate(batch):
        src_len = len(item['source'])
        tgt_len = len(item['target'])
        src_padded[i, :src_len] = item['source']
        tgt_padded[i, :tgt_len] = item['target']
    
    src_padding_mask = create_padding_mask(src_padded, pad_token)
    tgt_padding_mask = create_padding_mask(tgt_padded, pad_token)

    tgt_input = tgt_padded[:, :-1]
    tgt_output = tgt_padded[:, 1:]

    return {
        'source': src_padded,
        'target_input': tgt_input,
        'target_output': tgt_output,
        'src_padding_mask': src_padding_mask,
        'tgt_padding_mask': tgt_padding_mask,
        'src_lengths': torch.tensor([len(item['source']) for item in batch]),
        'tgt_lengths': torch.tensor([len(item['target']) for item in batch])
    }

def get_dataloader(
        data_dir: str,
        split: str,
        batch_size: int,
        max_seq_len: int = 512,
        num_workers: int = 4,
        shuffle: bool = True
        ) -> DataLoader:
    
    dataset = MTDataset(data_dir, split, max_seq_len, ddp_rank, ddp_world_size)
    if ddp_rank is not None:
        sampler = DistributedSampler(
            dataset,
            num_replicas=ddp_world_size,
            rank=ddp_rank,
            shuffle=shuffle
        )
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and shuffle),
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, pad_token=0),
        pin_memory=True,
        sampler=sampler
    )

if __name__ == '__main__':
    config = MTConfig()
    train_loader = get_dataloader(DATA_ROOT, 'train', config.batch_size, max_seq_len=config.max_seq_len)
    for batch in train_loader:
        print(batch['source'].shape, batch['target_input'].shape, batch['target_output'].shape)
        break
