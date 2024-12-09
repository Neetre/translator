from datasets import load_dataset
import tiktoken


dataset = load_dataset('wmt17', 'de-en')
tokenizer = tiktoken.get_encoding("cl100k_base")  # o200k_base

def tokenize_function(examples):
    german_texts = [example['de'] for example in examples['translation']]
    english_texts = [example['en'] for example in examples['translation']]
    
    return {
        'input_ids': [tokenizer.encode(text) for text in german_texts],
        'target_ids': [tokenizer.encode(text) for text in english_texts]
    }


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset['train'].column_names
)


print("First tokenized example:")
print("Input:", tokenized_dataset['train'][0]['input_ids'])
print("Target:", tokenized_dataset['train'][0]['target_ids'])