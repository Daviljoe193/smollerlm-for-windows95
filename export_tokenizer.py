"""
Exports a Hugging Face tokenizer to the .bin format expected by run.c
"""
import struct
import argparse
import os
from transformers import AutoTokenizer

def export_tokenizer(hf_path, output_path):
    print(f"Loading tokenizer from {hf_path}")
    tokenizer = AutoTokenizer.from_pretrained(hf_path)
    
    # SmollerLM (and GPT-2/Llama-3) use ByteLevel BPE. 
    # The C code expects the actual display bytes (e.g. " " instead of "Ġ").
    # We will iterate through all IDs and decode them to get the raw bytes.
    
    vocab_size = tokenizer.vocab_size
    # Handle added tokens (like <|im_start|>) which push the size beyond vocab_size
    if hasattr(tokenizer, 'len'):
        vocab_size = len(tokenizer)
    
    print(f"Vocab size: {vocab_size}")

    tokens = []
    scores = []

    for i in range(vocab_size):
        # 1. Get the string representation
        # We use decode() to handle the Ġ -> space conversion automatically
        # clean_up_tokenization_spaces=False ensures we keep explicit spaces
        text = tokenizer.decode([i], clean_up_tokenization_spaces=False)
        
        # 2. Edge case handling for special tokens
        # Sometimes decode() might return empty for special control tokens depending on config
        if i in tokenizer.all_special_ids:
            # Fallback to the raw token representation for special tokens (e.g. <|im_start|>)
            text = tokenizer.convert_ids_to_tokens(i)
        
        # 3. Convert to bytes for C storage
        b = text.encode('utf-8')
        tokens.append(b)
        
        # 4. Scores
        # run.c uses scores to determine merge priority during encoding (prompt processing).
        # HF BPE uses rank-based merging, not explicit scores.
        # A simple heuristic for run.c compatible encoding is length-based priority,
        # or we just set it to negative index to mimic rank (lower index = higher priority in strict BPE).
        # We'll use -index as a proxy for "rank" (common in BPE).
        scores.append(-(float(i)))

    # Calculate max_token_length for the C header
    max_token_length = max(len(t) for t in tokens)
    print(f"Max token length: {max_token_length}")

    print(f"Writing to {output_path}")
    with open(output_path, 'wb') as f:
        # Header: max_token_length (int)
        f.write(struct.pack("I", max_token_length))
        
        # Body: score (float), length (int), bytes (char array)
        for b, score in zip(tokens, scores):
            f.write(struct.pack("fI", score, len(b)))
            f.write(b)
            
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf", type=str, required=True, help="HuggingFace model path (e.g. mehmetkeremturkcan/SmollerLM2-10M-sftb)")
    parser.add_argument("-o", "--output", type=str, default="tokenizer.bin", help="Output filename")
    args = parser.parse_args()

    export_tokenizer(args.hf, args.output)
