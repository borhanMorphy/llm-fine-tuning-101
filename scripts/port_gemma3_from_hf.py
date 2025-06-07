from dataclasses import asdict
import json
import argparse
import os
import sys
from collections import OrderedDict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.getcwd())

from llm_lora_sft import Gemma3, ModelConfig


CHAT_TEMPLATE = """{{ bos_token }}<start_of_turn>user
{{ messages[0]['content'] | trim }}

{{ messages[1]['content'] | trim }}<end_of_turn>
<start_of_turn>model
{% if messages|length > 2 and messages[2]['content'] -%}
{{ messages[2]['content'] | trim }}<end_of_turn>
{%- endif %}
"""

MODEL_NAME_TO_HF_CHECKPOINT = {
    "gemma3-1b": "google/gemma-3-1b-it",
}


def port_weights_from_hf(checkpoint: str, model_name: str, target_path: str, max_seq_len: int):
    device = "cpu"
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device, dtype)
    model.config.sliding_window = model.config.sliding_window
    model.eval()
    batch_size: int = 1
    seq_len: int = 50

    input_ids = torch.randint(3, 1000, size=(batch_size, seq_len))
    attn_mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool).tril(
        diagonal=0
    )
    attn_weights = torch.zeros(batch_size, 1, seq_len, seq_len)
    attn_weights.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float("-inf"))

    original_st = model.state_dict()

    original_max_seq_len = 2**15 # 32k for 1b gemma3

    max_seq_len = min(original_max_seq_len, max_seq_len)

    config = ModelConfig(
        hidden_size=model.config.hidden_size,
        vocab_size=model.config.vocab_size,
        num_layers=model.config.num_hidden_layers,
        max_seq_len=max_seq_len,
        rope_global_theta=model.config.rope_theta,
        rope_local_theta=model.config.rope_local_base_freq,
        head_dim=model.config.head_dim,
        num_q_heads=model.config.num_attention_heads,
        num_kv_heads=model.config.num_key_value_heads,
        attention_bias=model.config.attention_bias,
        is_sliding_attention=[
            layer.self_attn.is_sliding for layer in model.model.layers
        ],
        window_size=model.config.sliding_window,
        padding_idx=model.config.pad_token_id,
        rms_norm_eps=model.config.rms_norm_eps,
        mlp_inter_hidden_size=model.config.intermediate_size,
        mlp_bias=False,
    )

    new_model = Gemma3(config).to(device, dtype)
    new_model.eval()

    new_st = new_model.state_dict()

    final_st = OrderedDict()

    for key, val in zip(new_st.keys(), original_st.values()):
        final_st[key] = val

    new_model.load_state_dict(final_st)

    with torch.no_grad():
        r1 = model(input_ids, attention_mask=attn_weights)
        r2 = new_model(input_ids)

    print((r1.logits - r2).abs().mean())

    ckpt = {
        "weights": new_model.state_dict(),
        "config": asdict(config),
    }

    file_path = os.path.join(target_path, f"{model_name}-it.ckpt")
    torch.save(ckpt, file_path)


def port_tokenizer_from_hf(checkpoint: str, model_name: str, target_path: str):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    vocab = tokenizer.get_vocab()

    new_vocab = {}

    for token, token_id in vocab.items():
        updated_token = token.replace("\u2581", " ")
        new_vocab[updated_token] = token_id

    special_tokens = {
        "bos_token": tokenizer.bos_token,
        "eos_token": tokenizer.eos_token,
        "pad_token": tokenizer.pad_token,
        "unk_token": tokenizer.unk_token,
    }

    tokenizer_config = {
        "chat_template": CHAT_TEMPLATE,
        "special_tokens": special_tokens,
        "vocab": new_vocab,
    }

    file_path = os.path.join(target_path, f"{model_name}-tokenizer.json")

    with open(file_path, "w") as foo:
        json.dump(
            tokenizer_config,
            foo,
            indent=4,
        )


def main(args):
    checkpoint = MODEL_NAME_TO_HF_CHECKPOINT[args.model_name]

    os.makedirs(args.target_path, exist_ok=True)

    port_tokenizer_from_hf(checkpoint, args.model_name, args.target_path)
    port_weights_from_hf(checkpoint, args.model_name, args.target_path, args.max_seq_len)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-name",
        "-m",
        type=str,
        required=True,
        choices=list(MODEL_NAME_TO_HF_CHECKPOINT.keys()),
    )
    ap.add_argument("--target-path", "-t", type=str, default="artifacts")
    ap.add_argument("--max-seq-len", "-msq", type=int, default=2**12)
    main(ap.parse_args())
