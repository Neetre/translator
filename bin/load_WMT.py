'''
Languages I want: de-en, ru-en, zh-en
'''

from datasets import load_dataset
import tiktoken


LANGUAGE_PAIRS = [
    ('de', 'en'),
    ('ru', 'en'),
    ('zh', 'en')
]
tokenizer = tiktoken.get_encoding("cl100k_base")  # o200k_base


def load_all_datasets():
    datasets = {}
    for src, tgt in LANGUAGE_PAIRS:
        try:
            datasets[f'{src}-{tgt}'] = load_dataset('wmt19', f'{src}-{tgt}')
        except Exception as e:
            print(f"Failed to load dataset for {src}-{tgt}: {e}")
    return datasets


def tokenize_function(examples, src_lang):
    source_texts = [example[src_lang] for example in examples['translation']]
    english_texts = [example['en'] for example in examples['translation']]

    source_texts = [f"<{src_lang}> " + text for text in source_texts]
    
    return {
        "source": tokenizer.encode_batch(source_texts),
        "target": tokenizer.encode_batch(english_texts)
    }


datasets = load_all_datasets()
tokenized_datasets = {}
for lang_pair, dataset in datasets.items():
    src_lang, tgt_lang = lang_pair.split('-')
    tokenized_datasets[lang_pair] = dataset.map(
        lambda examples: tokenize_function(examples, src_lang),
        batched=True
    )

print(tokenized_datasets.keys())
print(tokenized_datasets['de-en']['train'][0])
print(tokenized_datasets['ru-en']['train'][0])
print(tokenized_datasets['zh-en']['train'][0])
