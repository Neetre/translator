import os
import math
import time

from dataclasses import dataclass
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

import sys
with open(sys.argv[0]) as f:
    code = f.read()


from data_loader import get_dataloader, MTConfig, init_distributed
from load_WMT import DATA_ROOT
from mod_model import TransformerModel

ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process, device_type = init_distributed()


config = MTConfig()
@dataclass
class Hyper:
    num_vocab : int = config.vocab_size
    batch_size : int = config.batch_size
    sequence_length : int = config.max_seq_len
    num_iterations : int = 1480
    warmup_iters : int = 0
    cooldown_iters : int = 600
    weight_decay : float = 0.01
    val_loss_every : int = 225
    bert_score_every: int = 1000
    log_dir: str = "../training_logs"
args = Hyper()

os.makedirs(args.log_dir, exist_ok=True)
log_file = os.path.join(args.log_dir, "training_log.txt")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
assert total_batch_size % (config.batch_size * config.max_seq_len * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (config.batch_size * config.max_seq_len * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

torch.set_float32_matmul_precision('high')

train_loader = get_dataloader(DATA_ROOT, 'train', config.batch_size, max_seq_len=config.max_seq_len)
val_loader = get_dataloader(DATA_ROOT, 'val', config.batch_size, max_seq_len=config.max_seq_len)

model = TransformerModel(config)
model.to(device)
if config.use_compiler:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank], broadcast_buffers=False, gradient_as_bucket_view=True)
raw_model = model.module if ddp else model


max_lr = 5e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 1907

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

optimizer = raw_model.configure_optimizers(weight_decay=args.weight_decay, learning_rate=5e-4, device_type=device_type)
# have to fix the optimizer, because the function returns an optimizer and a scheduler

def train_step(batch, grad_accum=False):
    if ddp:
        model.require_backward_grad_sync = not grad_accum
    
    loss_accum = 0.0
    src, tgt = batch
    src, tgt = src.to(device), tgt.to(device)
    
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        logits = model(src, tgt)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1), ignore_index=0)
        
    if grad_accum:
        loss = loss / grad_accum_steps
    
    loss.backward()
    loss_accum += loss.detach()
    
    return loss_accum


def eval_step(batch):
    with torch.no_grad():
        model.eval()
        src, tgt = batch
        src, tgt = src.to(device), tgt.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits = model(src, tgt)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1), ignore_index=0)
        return loss


def save_checkpoint(step, val_loss=None):
    if master_process:
        checkpoint_path = os.path.join(args.log_dir, f"model_{step:05d}.pt")
        checkpoint = {
            'model': raw_model.state_dict(),
            'config': raw_model.config,
            'step': step,
            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"saved checkpoint to {checkpoint_path}")


def load_checkpoint(checkpoint_path):
    if master_process:
        checkpoint = torch.load(checkpoint_path)
        raw_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"loaded checkpoint from {checkpoint_path}")
        return checkpoint['step']
    return 0


def train():
    if ddp:
        dist.barrier()
    
    for step in range(args.num_iterations):
        t0 = time.time()

        if step % args.val_loss_every == 0:
            model.eval()
            val_loss_accum = 0.0
            val_steps = min(20, len(val_loader))
            for _ in range(val_steps):
                batch = next(iter(val_loader))
                loss = eval_step(batch)
                val_loss_accum += loss / val_steps
            
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            
            if master_process:
                print(f"step {step}: validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                
                if step > 0 and step % 5000 == 0:
                    save_checkpoint(step, val_loss_accum.item())

        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            batch = next(iter(train_loader))
            loss = train_step(batch, grad_accum=(micro_step < grad_accum_steps-1))
            loss_accum += loss
        
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        optimizer.step()
        
        if device_type == "cuda":
            torch.cuda.synchronize()

        t1 = time.time()
        dt = t1 - t0
        tokens_processed = config.batch_size * config.max_seq_len * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        
        if master_process:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | "
                  f"grad_norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

    save_checkpoint(args.num_iterations, val_loss_accum.item() if 'val_loss_accum' in locals() else None)
    
    if ddp:
        dist.barrier()
        destroy_process_group()


if __name__ == "__main__":
    train()
