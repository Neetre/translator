import os
import argparse
import multiprocessing as mp
import numpy as np
import tiktoken
from tqdm import tqdm
from typing import Dict, Tuple, List, Iterator, Union
import requests
import json
import gzip
from zipfile import ZipFile
from itertools import islice
from contextlib import contextmanager


import warnings
import transformers
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()


from transformers import T5Tokenizer

parser = argparse.ArgumentParser(description="Download and preprocess opus data")
parser.add_argument("-f", "--fine_tune", action="store_true", help="Fine-tune a model, in this case T5")
parser.add_argument("-d", "--data_dir", type=str, default="../data/opus", help="Directory to save the data")
parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each shard in tokens")
parser.add_argument("-n", "--max_pairs", type=int, default=10**7, help="Maximum number of sentence pairs to process")
parser.add_argument("-b", "--batch_size", type=int, default=1000, help="Number of sentences to process in each batch")
parser.add_argument("-m", "--max_seq_len", type=int, default=512, help="Maximum sequence length")
args = parser.parse_args()


DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', "opus")
max_seq_len = 1024
shard_size = 10**8
max_pairs = 10**7
batch_size = 1000


class DatasetIterator:
    def __init__(self, data_dir: str, batch_size: int):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.file_pairs = self._get_file_pairs()
    
    def _get_file_pairs(self) -> List[Tuple[str, str]]:
        files = os.listdir(self.data_dir)
        en_files = sorted([f for f in files if f.endswith('.en')])
        ko_files = sorted([f for f in files if f.endswith('.ko')])
        return list(zip(en_files, ko_files))

    @contextmanager
    def _open_file_pair(self, en_file: str, ko_file: str):
        with open(os.path.join(self.data_dir, en_file), 'r', encoding='utf-8') as en_f, \
             open(os.path.join(self.data_dir, ko_file), 'r', encoding='utf-8') as ko_f:
            yield en_f, ko_f

    def __iter__(self) -> Iterator[Dict[str, str]]:
        for en_file, ko_file in self.file_pairs:
            with self._open_file_pair(en_file, ko_file) as (en_f, ko_f):
                while True:
                    en_lines = list(islice(en_f, self.batch_size))
                    ko_lines = list(islice(ko_f, self.batch_size))

                    if not en_lines or not ko_lines:
                        break

                    for en_line, ko_line in zip(en_lines, ko_lines):
                        yield {"en": en_line.strip(), "ko": ko_line.strip()}


def get_data_dir(split):
    """Create and return language/split specific directory"""
    data_dir = os.path.join(DATA_ROOT, split)
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def download_file(url, output_path):
    """
    Download a file with progress bar
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, 'wb') as file, tqdm(
        desc=os.path.basename(output_path),
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)


def download_opus_data(corpus="OpenSubtitles", source_lang="en", target_lang="fi"):
    api_url = f"http://opus.nlpl.eu/opusapi/?corpus={corpus}&source={source_lang}&target={target_lang}&preprocessing=moses&version=latest"
    response = requests.get(api_url)
    data = json.loads(response.text)

    languages = source_lang + "-" + target_lang
    os.makedirs(f"{DATA_ROOT}_{languages}", exist_ok=True)

    for item in data["corpora"]:
        url = item["url"]
        filename = os.path.basename(url)
        output_path = os.path.join(f"{DATA_ROOT}_{languages}", filename)

        print(f"\nDownloading {filename}...")
        download_file(url, output_path)

        if filename.endswith('.xml.gz'):
            print(f"Extracting {filename}...")
            with gzip.open(output_path, 'rb') as f_in:
                with open(output_path[:-3], 'wb') as f_out:
                    f_out.write(f_in.read())
        elif filename.endswith('.zip'):
            print(f"Extracting {filename}...")
            with ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(f"{DATA_ROOT}_{languages}")

        print(f"Successfully processed {filename}")


def write_shard(src_tokens: np.ndarray, tgt_tokens: np.ndarray, split: str, shard_idx: int):
    data_dir = get_data_dir(split)

    src_path = os.path.join(data_dir, f"src_shard_{shard_idx:06d}.npy")
    tgt_path = os.path.join(data_dir, f"trg_shard_{shard_idx:06d}.npy")

    np.save(src_path, src_tokens)
    np.save(tgt_path, tgt_tokens)
    print(f"Saved shard {shard_idx} to {src_path} and {tgt_path} with {len(src_tokens)} tokens")


def process_batch_worker(batch, fine_tune, enc_name, max_seq_len):
    """Worker function for processing a batch of sentences."""
    if fine_tune:
        enc = T5Tokenizer.from_pretrained(enc_name)
        eot = enc.eos_token_id
    else:
        enc = tiktoken.get_encoding("cl100k_base")
        eot = enc._special_tokens["<|endoftext|>"]

    src_tokens_list = []
    tgt_tokens_list = []

    for doc in batch:
        if fine_tune:
            src_tokens = enc(
                doc["en"],
                padding=True,
                truncation=True,
                max_length=max_seq_len,
                return_tensors="pt"
            )["input_ids"].squeeze(0).numpy().tolist()
            tgt_tokens = enc(
                doc["ko"],
                padding=True,
                truncation=True,
                max_length=max_seq_len,
                return_tensors="pt"
            )["input_ids"].squeeze(0).numpy().tolist()
        else:
            src_tokens = [eot] + enc.encode_ordinary(doc["en"])
            tgt_tokens = [eot] + enc.encode_ordinary(doc["ko"])

        if len(src_tokens) <= max_seq_len and len(tgt_tokens) <= max_seq_len:
            src_tokens_list.append(np.array(src_tokens, dtype=np.uint32))
            tgt_tokens_list.append(np.array(tgt_tokens, dtype=np.uint32))

    return src_tokens_list, tgt_tokens_list


def preprocess_dataset():
    dataset = DatasetIterator(os.path.join(DATA_ROOT + "_en-ko"), args.batch_size)

    model_name = "t5-base" if args.fine_tune else "cl100k_base"
    nprocs = max(1, mp.cpu_count() - 2)
    print(f"Using {nprocs} processes")

    shard_idx = 0
    src_tokens_buffer = []
    tgt_tokens_buffer = []
    total_tokens = 0
    processed_pairs = 0

    with mp.Pool(nprocs) as pool:
        batch = []
        futures = []

        for doc in tqdm(dataset, desc="Processing documents", unit="pairs", total=args.max_pairs):
            if processed_pairs >= args.max_pairs:
                break

            batch.append(doc)

            if len(batch) >= args.batch_size:
                futures.append(pool.apply_async(process_batch_worker, (batch, args.fine_tune, model_name, args.max_seq_len)))
                batch = []

            while len(futures) >= nprocs:
                result = futures.pop(0).get()
                src_batch_tokens, tgt_batch_tokens = result

                src_tokens_buffer.extend(src_batch_tokens)
                tgt_tokens_buffer.extend(tgt_batch_tokens)
                total_tokens += sum(len(t) for t in src_batch_tokens) + sum(len(t) for t in tgt_batch_tokens)
                processed_pairs += len(src_batch_tokens)

                if total_tokens >= args.shard_size:
                    src_shard = np.concatenate(src_tokens_buffer)
                    tgt_shard = np.concatenate(tgt_tokens_buffer)

                    split = "val" if shard_idx == 0 else "train"
                    write_shard(src_shard, tgt_shard, split, shard_idx)

                    src_tokens_buffer = []
                    tgt_tokens_buffer = []
                    total_tokens = 0
                    shard_idx += 1

        if batch:
            futures.append(pool.apply_async(process_batch_worker, (batch, args.fine_tune, model_name, args.max_seq_len)))

        for future in futures:
            result = future.get()
            src_batch_tokens, tgt_batch_tokens = result
            src_tokens_buffer.extend(src_batch_tokens)
            tgt_tokens_buffer.extend(tgt_batch_tokens)

        if src_tokens_buffer:
            src_shard = np.concatenate(src_tokens_buffer)
            tgt_shard = np.concatenate(tgt_tokens_buffer)
            split = "val" if shard_idx == 0 else "train"
            write_shard(src_shard, tgt_shard, split, shard_idx)


if __name__ == "__main__":
    corpus = "CCMatrix"
    src_lang = "en"
    trc_lang = "ko"
    if not os.path.exists(os.path.join(DATA_ROOT + "_en-ko", "en-ko.txt.zip")):
        download_opus_data(corpus, src_lang, trc_lang)
    preprocess_dataset()
    print("All done!")
