'''
chinese dataset: shjwudp/chinese-c4 = 2M lines, IDEA-CCNL/laion2B-multi-chinese-subset = 22M lines, BAAI/IndustryCorpus2 = 826M lines, opencsg/chinese-fineweb-edu = 84M lines
english dataset: bookcorpus/bookcorpus = 74M lines
'''

import regex as re
import tiktoken
import json
import datasets
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')


def get_stats(ids: list, counts=None):
    """
    Get the frequency of each pair of ids in a list of ids.

    Args:
        ids (list): List of ids.
        counts (dict, optional): Dictionary of counts. Defaults to None.

    Returns:
        dict: Dictionary of counts.
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list, pair, idx: int):
    """
    Merge a pair of ids in a list of ids.

    Args:
        ids (list): List of ids.
        pair (): Pair of ids to merge.
        idx (int): Index to replace the pair with.

    Returns:
        list: New list of ids.
    """

    newids = []  # new list of ids
    i = 0
    while i < len(ids):
        #  if not at the very last position AND the pair matches, replace it
        if i < len(ids) -1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


class BaseTokenizer:

    def __init__(self) -> None:
        self.merges = {}
        self.pattern = ""
        self.special_tokens = {}
        self.vocab = self._build_vocab()

    def train(self, text, vocab_zise):
        pass

    def encode(self, text):
        pass

    def decode(self, ids):
        pass

    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")

        return vocab


GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class BytePairTokenizer(BaseTokenizer):
    def __init__(self, vocab_size=8000, max_token_length=8, min_frequency=3, max_entropy=5.0) -> None:
        super().__init__()
        assert vocab_size >= 256, "Vocab size must be at least 256"
        self.vocab_size = vocab_size
        self.num_merges = vocab_size - 256
        self.pattern = GPT4_SPLIT_PATTERN
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        self.max_token_length = max_token_length
        self.min_frequency = min_frequency
        self.max_entropy = max_entropy

    def train(self, text: str):

        ids = [list(tx.encode("utf-8")) for tx in re.findall(self.compiled_pattern, text)]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(self.num_merges):
            stats = {}

            for chunk_ids in ids:
                get_stats(chunk_ids, stats)

            valid_pairs = {
                pair: count for pair, count in stats.items()
                if (count >= self.min_frequency) and
                   (len(vocab[pair[0]]) + len(vocab[pair[1]]) <= self.max_token_length) and
                   (pair not in merges)
            }

            if not valid_pairs:
                break

            top_pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [merge(chunk_ids, top_pair, idx) for chunk_ids in ids]
            merges[top_pair] = idx
            vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]
            print(f"Merge {i+1}/{self.num_merges}: {top_pair} --> {idx}  | {vocab[idx]} had {stats[top_pair]} occurencies!!")

        self.merges = merges
        self.vocab = vocab

    def register_special_tokens(self, special_tokens: dict):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids: list):
        part_bytes = []

        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"Invalid token {idx}")

        tokens = b"".join(part_bytes)
        text = tokens.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes: bytes):
        ids = list(text_bytes)

        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text: str):
        text_chunks = re.findall(self.compiled_pattern, text)

        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)

        return ids

    def encode(self, text: str, allowed_special="none_raise"):
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"Invalid value for allowed_special: {allowed_special}")

        if not special:
            return self.encode_ordinary(text)

        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)

        ids = []

        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))

        return ids

    def save_merges(self):
        with open("../data/merges.json", "w") as file:
            db ={}
            mer = {str(k): v for k, v in self.merges.items()}
            vocab = {str(k): v.decode("utf-8", errors="replace") for k, v in self.vocab.items()}
            db["merges"] = mer
            db["vocab"] = vocab
            json.dump(db, file, indent=4)

    def load_merges(self):
        with open("../data/merges.json", "r") as file:
            db = json.load(file)
            merges = {eval(k): v for k, v in db["merges"].items()}
            vocab = {eval(k): v.encode("utf-8") for k, v in db["vocab"].items()}
            self.merges = merges
            self.vocab = vocab

    def view_tokenized_text(self, ids: list):
        for idx in ids:
            print(f"{self.vocab[idx].decode('utf-8', errors='replace')}: {self.vocab[idx]}")


class EnhancedBytePairTokenizer(BytePairTokenizer):
    def train(self, text: str):
        ids = [list(tx.encode("utf-8")) for tx in re.findall(self.compiled_pattern, text)]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        progress_bar = tqdm(range(self.num_merges), desc="Merging Tokens")
        
        for i in range(self.num_merges):
            stats = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)

            valid_pairs = {
                pair: count for pair, count in stats.items()
                if (count >= self.min_frequency) and
                   (len(vocab[pair[0]]) + len(vocab[pair[1]]) <= self.max_token_length) and
                   (pair not in merges)
            }

            if not valid_pairs:
                logging.info(f"No valid pairs found. Stopping at {i} merges.")
                break

            top_pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [merge(chunk_ids, top_pair, idx) for chunk_ids in ids]
            merges[top_pair] = idx
            vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]
            
            progress_bar.update(1)
            logging.debug(f"Merge {i+1}/{self.num_merges}: {top_pair} --> {idx} | {vocab[idx]} had {stats[top_pair]} occurrences")

        progress_bar.close()
        self.merges = merges
        self.vocab = vocab

    def analyze_tokenization(self, text: str):
        """
        Comprehensive tokenization analysis
        
        Returns:
            dict: Detailed tokenization metrics
        """
        ids = self.encode(text)
        original_bytes = text.encode('utf-8')
        
        return {
            'original_length': len(original_bytes),
            'token_length': len(ids),
            'compression_ratio': len(ids) / len(original_bytes),
            'unique_tokens': len(set(ids)),
            'token_distribution': self._get_token_distribution(ids)
        }

    def _get_token_distribution(self, ids):
        """
        Analyze distribution of tokens
        
        Returns:
            dict: Token frequency distribution
        """
        token_freq = {}
        for token in ids:
            token_freq[token] = token_freq.get(token, 0) + 1
        
        return sorted(token_freq.items(), key=lambda x: x[1], reverse=True)[:10]


def get_corpus(max_samples=100000, batch_size=1000):
    """Streaming corpus loader with more robust error handling"""
    try:
        bookcorpus = datasets.load_dataset("bookcorpus", split="train", streaming=True)
        text_chunks = []
        
        for i, sample in enumerate(bookcorpus):
            if i >= max_samples:
                break
            text_chunks.append(sample['text'])
            
            if len(text_chunks) >= batch_size:
                yield ' '.join(text_chunks)
                text_chunks = []
        
        if text_chunks:
            yield ' '.join(text_chunks)
    
    except Exception as e:
        logging.error(f"Error loading corpus: {e}")
        raise


def main():
    tokenizer = EnhancedBytePairTokenizer(
        vocab_size=800,  # Vocabulary size, number of tokens in the vocabulary
        max_token_length=4, # Maximum token length, referes to the number of bytes in a token
        min_frequency=5  # Minimum frequency of a token to be considered for merging
    )
    
    for text_chunk in get_corpus(max_samples=10000):
        tokenizer.train(text_chunk)

    tokenizer.save_merges()
    
    # Example
    sample_text = "Hello, this is a sample text for tokenization analysis."
    analysis = tokenizer.analyze_tokenization(sample_text)
    logging.info(f"Tokenization Analysis: {json.dumps(analysis, indent=2)}")
    print(tokenizer.view_tokenized_text(tokenizer.encode(sample_text)))


if __name__ == "__main__":
    main()