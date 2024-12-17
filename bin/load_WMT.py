'''
Languages I want: de-en, ru-en, zh-en
'''
import os
import argparse
import multiprocessing as mp
import numpy as np
from datasets import load_dataset
import tiktoken
from tqdm import tqdm
from functools import partial
from typing import Dict, Tuple


parser = argparse.ArgumentParser(description="Download and preprocess WMT19 data")
parser.add_argument("-d", "--data_dir", type=str, default="../data/WMT19", help="Directory to save the data")
parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each shard in tokens")
parser.add_argument("-m", "--max_pairs", type=int, default=10**6, help="Maximum number of sentence pairs to process")
args = parser.parse_args()


LANGUAGE_PAIRS = ['de-en', 'ru-en', 'zh-en']
BASE_DIR = os.path.dirname(__file__)
DATA_ROOT = os.path.join(BASE_DIR, "..", "data")
max_seq_len = 1024


def get_data_dir(lang_pair, split):
    """Create and return language/split specific directory"""
    data_dir = os.path.join(DATA_ROOT, lang_pair, split)
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


enc = tiktoken.get_encoding("cl100k_base")  # o200k_base - cl100k_base
# eot = enc._special_tokens['<|endoftext|>'] # end of text token
# vocab_size = enc.vocab_size # 100257, it is not divisible by 128, so might be better 100352


# Handle both source and target languages
def tokenize(doc: Dict, lang_pair: str) -> Tuple[np.ndarray, np.ndarray]:
    lang_src, lang_trg = lang_pair.split("-")

    src_tokens = enc.encode_ordinary(doc["translation"][lang_src])
    tgt_tokens = enc.encode_ordinary(doc["translation"][lang_trg])

    src_tokens_np = np.array(src_tokens)
    tgt_tokens_np = np.array(tgt_tokens)

    assert (0 <= src_tokens_np).all() and (src_tokens_np < 2**32).all(), "token dictionary too large for uint32"
    assert (0 <= tgt_tokens_np).all() and (tgt_tokens_np < 2**32).all(), "token dictionary too large for uint32"

    src_tokens_np_uint32 = src_tokens_np.astype(np.uint32)
    tgt_tokens_np_uint32 = tgt_tokens_np.astype(np.uint32)

    return src_tokens_np_uint32, tgt_tokens_np_uint32


def write_shard(src_tokens: np.ndarray, tgt_tokens: np.ndarray, lang_pair: str, split: str, shard_idx: int):
    data_dir = get_data_dir(lang_pair, split)

    src_path = os.path.join(data_dir, f"src_shard_{shard_idx:06d}.npy")
    tgt_path = os.path.join(data_dir, f"trg_shard_{shard_idx:06d}.npy")

    np.save(src_path, src_tokens)
    np.save(tgt_path, tgt_tokens)
    print(f"Saved shard {shard_idx} to {src_path} and {tgt_path} with {len(src_tokens)} tokens")


def preprocess_dataset(lang_pair: str):
    dataset = load_dataset('wmt19', lang_pair, split='train', streaming=True)

    shard_idx = 0
    src_tokens_buffer = []
    tgt_tokens_buffer = []
    total_tokens = 0
    processed_pairs = 0

    nprocs = max(1, mp.cpu_count() - 2)
    print(f"Using {nprocs} processes")

    with mp.Pool(max(1, mp.cpu_count() - 2)) as pool:
        partial_tokenizer = partial(tokenize, lang_pair=lang_pair)

        for src_toks, tgt_toks in tqdm(pool.imap(partial_tokenizer, dataset, chunksize=16),
                                     unit="tokens",
                                     desc=f"Processing {lang_pair}", ):
            
            if processed_pairs >= args.max_pairs:
                break

            processed_pairs += 1

            if len(src_toks) > max_seq_len or len(tgt_toks) > max_seq_len:
                continue

            src_tokens_buffer.append(src_toks)
            tgt_tokens_buffer.append(tgt_toks)
            total_tokens += len(src_toks) + len(tgt_toks)

            if total_tokens >= args.shard_size:
                src_shard = np.concatenate(src_tokens_buffer)
                tgt_shard = np.concatenate(tgt_tokens_buffer)

                # First shard goes to validation
                split = "val" if shard_idx == 0 else "train"
                write_shard(src_shard, tgt_shard, lang_pair, split, shard_idx)

                src_tokens_buffer = []
                tgt_tokens_buffer = []
                total_tokens = 0
                shard_idx += 1

        # Write final shard if there's remaining data
        if src_tokens_buffer:
            src_shard = np.concatenate(src_tokens_buffer)
            tgt_shard = np.concatenate(tgt_tokens_buffer)
            split = "val" if shard_idx == 0 else "train"
            write_shard(src_shard, tgt_shard, lang_pair, split, shard_idx)


if __name__ == "__main__":
    for lang_pair in LANGUAGE_PAIRS:
        preprocess_dataset(lang_pair)
