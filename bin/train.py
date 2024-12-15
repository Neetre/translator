import os
import uuid
from pathlib import Path

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import sys
with open(sys.argv[0]) as f:
    code = f.read()

from data_loader import MTDataset, MTConfig
from load_WMT import DATA_ROOT, LANGUAGE_PAIRS
from mod_model import TransformerModel

config = MTConfig()
@dataclass
class Hyper:
    num_vocab : int = config.vocab_size
    batch_size : int = config.batch_size
    sequence_length : int = config.max_seq_len
    num_iterations : int = 1480
    warmup_iters : int = 0
    cooldown_iters : int = 600
    weight_decay : float = 0
    val_loss_every : int = 125
args = Hyper()


def init_distributed():
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    assert torch.cuda.is_available()
    device = torch.device(f'cuda:{ddp_local_rank}')
    torch.cuda.set_device(device)
    print(f'using device: {device}')
    dist.init_process_group(backend='nccl', device_id=device)
    dist.barrier()
    master_process = (ddp_rank == 0)
    return ddp_rank, ddp_local_rank, ddp_world_size, device, master_process

ddp_rank, ddp_local_rank, ddp_world_size, device, master_process = init_distributed()


logfile = None
if master_process:
    run_id = uuid.uuid4()
    Path('logs').mkdir(exist_ok=True)
    logfile = Path('logs') / f'{run_id}.txt'
    print(logfile.stem)
    with logfile.open('w') as f:
        print(code, file=f)
        print('=' * 100, file=f)
def print0(s, logonly=False):
    if master_process:
        with logfile.open('a') as f:
            if not logonly:
                print(s)
            print(s, file=f)


print0(f'Running python {sys.version}')
print0(f'Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:')
import subprocess
result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
print0(f'{result.stdout}', logonly=True)
print0('='*100, logonly=True)


train_loader = MTDataset(DATA_ROOT, 'train', config.max_seq_len, ddp_rank, ddp_world_size)
val_loader = MTDataset(DATA_ROOT, 'val', config.max_seq_len, ddp_rank, ddp_world_size)

model = TransformerModel(config).to(device)