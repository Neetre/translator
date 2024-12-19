import os
import argparse
import multiprocessing as mp
import numpy as np
import tiktoken
from tqdm import tqdm
from typing import Dict, Tuple
import requests
import json
import gzip
from zipfile import ZipFile


parser = argparse.ArgumentParser(description="Download and preprocess opus data")
parser.add_argument("-d", "--data_dir", type=str, default="../data/opus", help="Directory to save the data")
parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each shard in tokens")
parser.add_argument("-m", "--max_pairs", type=int, default=10**7, help="Maximum number of sentence pairs to process")
args = parser.parse_args()


BASE_DIR = os.path.dirname(__file__)
DATA_ROOT = os.path.join(BASE_DIR, "..", "data", "opus_data")
max_seq_len = 1024


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


def get_dataset():
    data_dir = os.path.join(DATA_ROOT+"_en-ko")
    dataset = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".en") or filename.endswith(".ko"):
            lang = filename.split(".")[-1]
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as file:
                lines = file.readlines()

            for i, line in enumerate(lines):
                if lang == "en":
                    if i >= len(dataset):
                        dataset.append({"en": line.strip(), "ko": ""})
                    else:
                        dataset[i]["en"] = line.strip()
                elif lang == "ko":
                    if i >= len(dataset):
                        dataset.append({"en": "", "ko": line.strip()})
                    else:
                        dataset[i]["ko"] = line.strip()

    for item in dataset:
        yield item


enc = tiktoken.get_encoding("cl100k_base")


def tokenize(doc: Dict) -> Tuple[np.ndarray, np.ndarray]:

    src_tokens = enc.encode_ordinary(doc["en"])  # english
    tgt_tokens = enc.encode_ordinary(doc["ko"])  # korean

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


def preprocess_dataset():
    dataset = get_dataset()

    shard_idx = 0
    src_tokens_buffer = []
    tgt_tokens_buffer = []
    total_tokens = 0
    processed_pairs = 0

    nprocs = max(1, mp.cpu_count() - 2)
    print(f"Using {nprocs} processes")

    with mp.Pool(nprocs) as pool:
        for src_toks, tgt_toks in tqdm(pool.imap(tokenize, dataset, chunksize=16), unit="tokens", desc="Processing"):

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
                write_shard(src_shard, tgt_shard, split, shard_idx)

                src_tokens_buffer = []
                tgt_tokens_buffer = []
                total_tokens = 0
                shard_idx += 1

        # Write final shard if there's remaining data
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
