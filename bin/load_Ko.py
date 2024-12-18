import os
import argparse
import multiprocessing as mp
import numpy as np
import tiktoken
from tqdm import tqdm
from typing import Dict, Tuple
from Korpora import Korpora


parser = argparse.ArgumentParser(description="Download and preprocess WMT19 data")
parser.add_argument("-d", "--data_dir", type=str, default="../data/WMT19", help="Directory to save the data")
parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each shard in tokens")
parser.add_argument("-m", "--max_pairs", type=int, default=10**7, help="Maximum number of sentence pairs to process")
args = parser.parse_args()


BASE_DIR = os.path.dirname(__file__)
DATA_ROOT = os.path.join(BASE_DIR, "..", "data")
max_seq_len = 1024


def get_data_dir(split):
    """Create and return language/split specific directory"""
    data_dir = os.path.join(DATA_ROOT, split)
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def get_dataset():
    corpus = Korpora.load("aihub_translation", root_dir="../data/Korpora")
    for text in corpus.get_all_texts():
        yield text  # memory efficient


enc = tiktoken.get_encoding("cl100k_base")


def tokenize(doc: Dict) -> Tuple[np.ndarray, np.ndarray]:

    src_tokens = enc.encode_ordinary(doc.text)  # korean
    tgt_tokens = enc.encode_ordinary(doc.pair)  # english

    src_tokens_np = np.array(src_tokens)
    tgt_tokens_np = np.array(tgt_tokens)

    assert (0 <= src_tokens_np).all() and (src_tokens_np < 2**32).all(), "token dictionary too large for uint32"
    assert (0 <= tgt_tokens_np).all() and (tgt_tokens_np < 2**32).all(), "token dictionary too large for uint32"

    src_tokens_np_uint32 = src_tokens_np.astype(np.uint32)
    tgt_tokens_np_uint32 = tgt_tokens_np.astype(np.uint32)

    return src_tokens_np_uint32, tgt_tokens_np_uint32


def write_shard(src_tokens: np.ndarray, tgt_tokens: np.ndarray, split: str, shard_idx: int):
    data_dir = get_data_dir(split)

    src_path = os.path.join(data_dir, f"src_shard_{shard_idx:06d}.npy")
    tgt_path = os.path.join(data_dir, f"trg_shard_{shard_idx:06d}.npy")

    np.save(src_path, src_tokens)
    np.save(tgt_path, tgt_tokens)
    print(f"Saved shard {shard_idx} to {src_path} and {tgt_path} with {len(src_tokens)} tokens")

