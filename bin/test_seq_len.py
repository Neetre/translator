import tiktoken

def analyze_token_length():
    enc = tiktoken.get_encoding("cl100k_base")
    
    # Sample text with 100 words
    text = "The quick brown fox jumps over the lazy dog. " * 10
    
    tokens = enc.encode(text)
    words = text.split()
    
    print(f"Words: {len(words)}")
    print(f"Tokens: {len(tokens)}")
    print(f"Tokens per word: {len(tokens)/len(words):.2f}")
    print(f"Estimated words in 1024 sequence: {1024/(len(tokens)/len(words)):.0f}")

analyze_token_length()