import os
import gzip
import shutil
import struct
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from model import ModelArgs, Transformer

# -----------------------------------------------------------------------------
# common utilities

def serialize_fp32(file, tensor):
    """ writes one fp32 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)

def serialize_int8(file, tensor):
    """ writes one int8 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).numpy().astype(np.int8)
    b = struct.pack(f'{len(d)}b', *d)
    file.write(b)

def quantize_q80(w, group_size):
    """
    takes a tensor and returns the Q8_0 quantized version
    i.e. symmetric quantization into int8, range [-127,127]
    """
    assert w.numel() % group_size == 0
    ori_shape = w.shape
    w = w.float() # convert to float32
    w = w.reshape(-1, group_size)
    # find the max in each group
    wmax = torch.abs(w).max(dim=1).values
    # calculate the scaling factor such that float = quant * scale
    scale = wmax / 127.0
    # scale into range [-127, 127]
    quant = w / scale[:,None]
    # round to nearest integer
    int8val = torch.round(quant).to(torch.int8)
    # dequantize by rescaling
    fp32val = (int8val.float() * scale[:,None]).view(-1)
    fp32valr = fp32val.reshape(-1, group_size)
    # calculate the max error in each group
    err = torch.abs(fp32valr - w).max(dim=1).values
    # find the max error across all groups
    maxerr = err.max().item()
    return int8val, scale, maxerr

# -----------------------------------------------------------------------------
# export utilities

def version2_export(model, filepath, group_size=64):
    version = 2

    # Validation for quantization group size
    # We must ensure group_size divides the embedding dim, but also the head_dim projection sizes
    # Since this model has disparate sizes (60 vs 576), we iterate to find a common divisor
    # that is smaller than or equal to the requested group_size.
    def find_divisor(val, start_group):
        g = start_group
        while val % g != 0:
            g //= 2
        return g
    
    # Check embedding dim
    group_size = find_divisor(model.params.dim, group_size)
    # Check Attention projection dim (n_heads * head_dim)
    attn_dim = model.layers[0].attention.wq.weight.shape[0]
    group_size = find_divisor(attn_dim, group_size)
    
    print(f"Final quantization group size: {group_size}")

    weights = [
        model.tok_embeddings.weight,
        *[layer.attention.wq.weight for layer in model.layers],
        *[layer.attention.wk.weight for layer in model.layers],
        *[layer.attention.wv.weight for layer in model.layers],
        *[layer.attention.wo.weight for layer in model.layers],
        *[layer.feed_forward.w1.weight for layer in model.layers],
        *[layer.feed_forward.w2.weight for layer in model.layers],
        *[layer.feed_forward.w3.weight for layer in model.layers],
    ]
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    if not shared_classifier:
        weights.append(model.output.weight)
    
    for i, w in enumerate(weights):
        assert w.numel() % group_size == 0, f"weight {i} shape {w.shape} numel {w.numel()} not divisible by group_size {group_size}"

    # write
    out_file = open(filepath, 'wb')
    out_file.write(struct.pack('I', 0x616b3432)) # magic "ak42"
    out_file.write(struct.pack('i', version))
    
    p = model.params
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    
    # Get head_dim explicitly. If not set in params (it usually isn't in ModelArgs), infer from weights
    # wq shape is [n_heads * head_dim, dim]
    head_dim = model.layers[0].attention.wq.weight.shape[0] // p.n_heads
    
    # HEADER UPDATE: Added head_dim as the 8th integer
    header = struct.pack('iiiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                                    n_kv_heads, p.vocab_size, p.max_seq_len, head_dim)
    out_file.write(header)
    
    out_file.write(struct.pack('B', int(shared_classifier)))
    out_file.write(struct.pack('i', group_size)) 
    pad = 256 - out_file.tell() 
    assert pad >= 0
    out_file.write(b'\0' * pad)

    # Write fp32 params (norms)
    for layer in model.layers: serialize_fp32(out_file, layer.attention_norm.weight)
    for layer in model.layers: serialize_fp32(out_file, layer.ffn_norm.weight)
    serialize_fp32(out_file, model.norm.weight) 

    # Write quantized params
    ew = []
    for i, w in enumerate(weights):
        q, s, err = quantize_q80(w, group_size)
        serialize_int8(out_file, q)
        serialize_fp32(out_file, s)
        ew.append((err, w.shape))
        if i % 10 == 0: print(f"{i+1}/{len(weights)} quantized {tuple(w.shape)} err {err:.2e}")

    ew.sort(reverse=True)
    print(f"max quantization group error: {ew[0][0]}")
    out_file.close()
    print(f"wrote {filepath}")

# -----------------------------------------------------------------------------
# Load / import functions

def load_hf_model(model_path):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("Error: transformers package is required to load huggingface models")
        return None

    hf_model = AutoModelForCausalLM.from_pretrained(model_path)
    hf_dict = hf_model.state_dict()

    config = ModelArgs()
    config.dim = hf_model.config.hidden_size
    config.n_layers = hf_model.config.num_hidden_layers
    config.n_heads = hf_model.config.num_attention_heads
    config.n_kv_heads = hf_model.config.num_key_value_heads
    config.vocab_size = hf_model.config.vocab_size
    config.hidden_dim = hf_model.config.intermediate_size
    config.norm_eps = hf_model.config.rms_norm_eps
    config.max_seq_len = hf_model.config.max_position_embeddings
    
    # Crucial for SmollerLM: Head dim might not be dim // n_heads
    # We grab it from config if available (some HF configs have it), or default calculation
    if hasattr(hf_model.config, 'head_dim'):
        head_dim = hf_model.config.head_dim
    else:
        # Fallback to standard Llama
        head_dim = config.dim // config.n_heads

    print(f"Loaded config: dim={config.dim}, head_dim={head_dim}, n_heads={config.n_heads}")

    model = Transformer(config)

    # Force the shapes to match HF (since ModelArgs defaults to standard Llama shapes)
    # We recreate the layers with correct projection sizes if they differ
    for layer in model.layers:
        layer.attention.wq = nn.Linear(config.dim, config.n_heads * head_dim, bias=False)
        layer.attention.wk = nn.Linear(config.dim, config.n_kv_heads * head_dim, bias=False)
        layer.attention.wv = nn.Linear(config.dim, config.n_kv_heads * head_dim, bias=False)
        layer.attention.wo = nn.Linear(config.n_heads * head_dim, config.dim, bias=False)

    model.tok_embeddings.weight = nn.Parameter(hf_dict['model.embed_tokens.weight'])
    model.norm.weight = nn.Parameter(hf_dict['model.norm.weight'])

    # Permute helper that accepts explicit dimensions to handle SmollerLM shapes
    def permute_reverse(w, n_heads=config.n_heads, dim1=config.dim, dim2=config.dim):
        # w shape is [dim1, dim2] usually
        # We want to un-permute the rotary embedding layout
        # dim1 is the output dimension (e.g. 576), dim2 is input (e.g. 60)
        return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

    for layer in model.layers:
        i = layer.layer_id
        layer.attention_norm.weight = nn.Parameter(hf_dict[f'model.layers.{i}.input_layernorm.weight'])
        
        # Q Projection: Output dim is n_heads * head_dim
        q_w = hf_dict[f'model.layers.{i}.self_attn.q_proj.weight']
        layer.attention.wq.weight = nn.Parameter(permute_reverse(q_w, n_heads=config.n_heads, dim1=config.n_heads * head_dim, dim2=config.dim))
        
        # K Projection: Output dim is n_kv_heads * head_dim
        k_w = hf_dict[f'model.layers.{i}.self_attn.k_proj.weight']
        layer.attention.wk.weight = nn.Parameter(permute_reverse(k_w, n_heads=config.n_kv_heads, dim1=config.n_kv_heads * head_dim, dim2=config.dim))
        
        layer.attention.wv.weight = nn.Parameter(hf_dict[f'model.layers.{i}.self_attn.v_proj.weight'])
        layer.attention.wo.weight = nn.Parameter(hf_dict[f'model.layers.{i}.self_attn.o_proj.weight'])
        layer.ffn_norm.weight = nn.Parameter(hf_dict[f'model.layers.{i}.post_attention_layernorm.weight'])
        layer.feed_forward.w1.weight = nn.Parameter(hf_dict[f'model.layers.{i}.mlp.gate_proj.weight'])
        layer.feed_forward.w2.weight = nn.Parameter(hf_dict[f'model.layers.{i}.mlp.down_proj.weight'])
        layer.feed_forward.w3.weight = nn.Parameter(hf_dict[f'model.layers.{i}.mlp.up_proj.weight'])

    model.output.weight = nn.Parameter(hf_dict['lm_head.weight'])
    model.eval()
    return model

# -----------------------------------------------------------------------------
# CLI entrypoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="the output filepath")
    parser.add_argument("--version", default=2, type=int, help="the version to export with")
    parser.add_argument("--hf", type=str, help="huggingface model path", required=True)
    args = parser.parse_args()

    model = load_hf_model(args.hf)
    if model is None: parser.error("Can't load input model!")
    
    # Only supporting version 2 (INT8) for this specific custom fix
    if args.version == 2:
        version2_export(model, args.filepath)
    else:
        print("This custom script only supports version 2 export for SmollerLM compatibility.")
