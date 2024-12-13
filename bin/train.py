from dataclasses import dataclass
from torch.utils.data import DataLoader

from mod_model import TransformerModel


@dataclass
class GPTConfig:
    vocab_size : int = # to choose based on the tokenizer
    num_layers : int = 12
    num_heads : int = 6 # head dim 128
    model_dim : int = 768


train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None)
    )