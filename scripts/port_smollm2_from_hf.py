from dataclasses import asdict
import json
import argparse
import os
import sys
from collections import OrderedDict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.getcwd())

from slm_lora_sft import SmolLM2, ModelConfig


CHAT_TEMPLATE = """<|im_start|>system
{{ messages[0]['content'] | trim }}
<|im_end|>
<|im_start|>user
{{ messages[1]['content'] | trim }}
<|im_end|>
<|im_start|>assistant
{% if messages|length > 2 and messages[2]['content'] -%}
{{ messages[2]['content'] | trim }}
<|im_end|>
{%- endif %}
"""

MODEL_NAME_TO_HF_CHECKPOINT = {
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
}


def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


BYTE_DECODER = {v: k for k, v in bytes_to_unicode().items()}


def decode_unicode_to_text(unicode_text: str) -> str:
    byte_seq = bytes([BYTE_DECODER[ch] for ch in unicode_text])
    return byte_seq.decode("utf-8")


def port_weights_from_hf(checkpoint: str, model_name: str, target_path: str):
    device = "cpu"
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device, dtype)
    model.eval()
    batch_size: int = 1
    seq_len: int = 10

    input_ids = torch.randint(3, 1000, size=(batch_size, seq_len))
    attn_mask = torch.ones(batch_size, seq_len, seq_len, dtype=torch.bool).tril(
        diagonal=0
    )
    attn_weights = torch.zeros(batch_size, 1, seq_len, seq_len)
    attn_weights.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float("-inf"))

    original_st = model.state_dict()

    config = ModelConfig(
        hidden_size=model.config.hidden_size,
        vocab_size=model.config.vocab_size,
        num_layers=model.config.num_hidden_layers,
        max_seq_len=model.config.max_position_embeddings,
        rope_theta=model.config.rope_theta,
        num_q_heads=model.config.num_attention_heads,
        num_kv_heads=model.config.num_key_value_heads,
        attention_bias=model.config.attention_bias,
        padding_idx=model.config.pad_token_id,
        rms_norm_eps=model.config.rms_norm_eps,
        mlp_inter_hidden_size=model.config.intermediate_size,
        mlp_bias=model.config.mlp_bias,
    )

    new_model = SmolLM2(config).to(device, dtype)
    new_model.eval()

    new_st = new_model.state_dict()

    final_st = OrderedDict()

    for key, val in zip(new_st.keys(), original_st.values()):
        if len(val.shape) == 1:
            val = val.unsqueeze(0).unsqueeze(0)
        final_st[key] = val

    new_model.load_state_dict(final_st)

    with torch.no_grad():
        r1 = model(input_ids, attention_mask=attn_weights)
        r2 = new_model(input_ids, attention_mask=attn_mask)

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
        try:
            updated_token = decode_unicode_to_text(token)
        except Exception as e:
            updated_token = token

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
    port_weights_from_hf(checkpoint, args.model_name, args.target_path)


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
    main(ap.parse_args())
