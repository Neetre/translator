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


LANGUAGE_PAIRS = [
    ('de-en'),
    ('ru-en'),
    ('zh-en')
]

parser = argparse.ArgumentParser(description="Download and preprocess WMT19 data")
parser.add_argument("-d", "--data_dir", type=str, default="../data/WMT19", help="Directory to save the data")
parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each shard in tokens")
args = parser.parse_args()

local_dir = "../data/WMT19"
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)


enc = tiktoken.get_encoding("cl100k_base")  # o200k_base - cl100k_base
eot = enc._special_tokens['<|endoftext|>'] # end of text token
vocab_size = enc.vocab_size # 100257, it is not divisible by 128, so might be better 100352

def load_single_dataset(name_dataset, lang_pair, split, streaming=True):
    """Load dataset with memory-efficient streaming option"""
    dataset = load_dataset(
        name_dataset, 
        lang_pair, 
        split=split,
        streaming=streaming  # Enable streaming mode
    )
    return dataset

'''
lang_pair = 'de-en'
lang_src, lang_trg = lang_pair.split("-")
dataset = load_single_dataset('wmt19', 'de-en', split='train')
for example in dataset:
    print(example)
    encoded_src = enc.encode_ordinary(example['translation'][lang_src])
    encoded_trg = enc.encode_ordinary(example['translation'][lang_trg])
    print(encoded_src)
    print(encoded_trg)
    print(enc.decode(encoded_src))
    print(enc.decode(encoded_trg))
    break
'''

# Handle both source and target languages
def tokenize(doc, lang_pair):
    if isinstance(doc, dict):
        src_tokens = [eot]
        tgt_tokens = [eot]
        src_tokens.extend(enc.encode_ordinary(doc["translation"][lang_pair[0]]))
        tgt_tokens.extend(enc.encode_ordinary(doc["translation"][lang_pair[1]]))

        src_tokens_np = np.array(src_tokens)
        tgt_tokens_np = np.array(tgt_tokens)

        assert (0 <= src_tokens_np).all() and (src_tokens_np < 2**32).all(), "token dictionary too large for uint32"
        assert (0 <= tgt_tokens_np).all() and (tgt_tokens_np < 2**32).all(), "token dictionary too large for uint32"

        src_tokens_np_uint32 = src_tokens_np.astype(np.uint32)
        tgt_tokens_np_uint32 = tgt_tokens_np.astype(np.uint32)
        
        return src_tokens_np_uint32, tgt_tokens_np_uint32
    else:
        tokens = [eot]
        tokens.extend(enc.encode_ordinary(doc))
        tokens_np = np.array(tokens)
        assert (0 <= tokens_np).all() and (tokens_np < 2**32).all(), "token dictionary too large for uint32"
        tokens_np_uint32 = tokens_np.astype(np.uint32)
        return tokens_np_uint32


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)
    print(f"Saved {filename}")


nprocs = max(1, mp.cpu_count() - 2)
print(f"Using {nprocs} processes")
with mp.Pool(nprocs) as pool:
    for lang_pair in LANGUAGE_PAIRS:
        dataset = load_single_dataset('wmt19', lang_pair, split='train', streaming=True)
        shard_idx = 0
        all_tokens_np = np.empty((args.shard_size,), dtype=np.uint32)
        token_count = 0
        progress_bar = 0
        for dict_tokens in pool.imap(tokenize, dataset, chunksize=16):
            token_len = len(dict_tokens[0]) + len(dict_tokens[1])
            if token_count + token_len > args.shard_size:
                all_tokens_np[token_count:token_count+token_len] = dict_tokens
                token_count += token_len
                if progress_bar is None:
                    progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_idx}")
                progress_bar.update(token_count)
            else:
                split = "val" if shard_idx == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"{lang_pair[0]}_{lang_pair[1]}_{split}_{shard_idx:06d}.npy")
                remainder = args.shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = dict_tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_idx += 1
                progress_bar = None
                all_tokens_np[0: token_len - remainder] = dict_tokens[remainder:]
                token_count = token_len - remainder
        
        if token_count != 0:
            split = "val" if shard_idx == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"{lang_pair[0]}_{lang_pair[1]}_{split}_{shard_idx:06d}.npy")
            write_datafile(filename, all_tokens_np[:token_count])
