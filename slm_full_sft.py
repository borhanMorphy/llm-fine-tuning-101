from typing import Literal, List, Dict, Tuple, Optional, Generator
import argparse
import os
import json
from jinja2 import Template
from dataclasses import dataclass, asdict
from copy import deepcopy
from tqdm import tqdm
from functools import partial
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, LongTensor, BoolTensor
from torch.utils.data import Dataset, DataLoader

## ------------------- Data ------------------- ##


@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


class InstructDataset(Dataset):
    def __init__(self, data: List[List[Message]], tokenizer):
        super().__init__()
        self._data = data
        self._tokenizer = tokenizer

    def get_sample(self, index: int) -> str:
        messages = deepcopy(self._data[index])
        input_raw: str = self._tokenizer.apply_chat_template(messages=messages)
        return input_raw

    def __getitem__(self, index: int) -> str | List[int]:
        input_raw: str = self.get_sample(index)

        input_tokens: List[int] = self._tokenizer(input_raw)
        return input_tokens

    def __len__(self) -> int:
        return len(self._data)

    @classmethod
    def from_jsonl(cls, jsonl_file_path: str, *args, **kwargs):
        with open(jsonl_file_path, "r") as foo:
            raw_data = foo.read().split("\n")

        data = []
        for sample_json in raw_data:
            messages = []
            for message in json.loads(sample_json):
                messages.append(Message(**message))
            data.append(messages)

        return cls(data, *args, **kwargs)


def dynamic_collate_fn(all_token_ids, pad_token_id: int = 0, ignore_index: int = -100):
    max_seq_len = max([len(token_ids) for token_ids in all_token_ids]) - 1
    batch_input_ids = []
    batch_targets = []
    batch_attn_mask = []

    for token_ids in all_token_ids:
        token_ids = torch.tensor(token_ids, dtype=torch.long)

        input_ids = token_ids[:-1]
        targets = token_ids[1:]

        seq_len = input_ids.shape[0]

        num_pads = max_seq_len - seq_len

        input_ids = F.pad(input_ids, (0, num_pads), value=pad_token_id)
        targets = F.pad(targets, (0, num_pads), value=ignore_index)
        attn_mask = torch.ones(1, max_seq_len, max_seq_len, dtype=torch.bool).tril(
            diagonal=0
        )

        attn_mask[:, seq_len:, :] = False

        batch_input_ids.append(input_ids)
        batch_targets.append(targets)
        batch_attn_mask.append(attn_mask)

    batch_input_ids = torch.stack(batch_input_ids)
    batch_targets = torch.stack(batch_targets)
    batch_attn_mask = torch.stack(batch_attn_mask)

    return batch_input_ids, batch_targets, batch_attn_mask


## ------------------- Tokenization ------------------- ##


class _TreeNode:
    def __init__(self):
        self.children = {}
        self.is_token = False


class _VocabTree:
    def __init__(self):
        self.root = _TreeNode()

    def add_token(self, token: str):
        node = self.root

        for ch in token:
            if ch not in node.children:
                node.children[ch] = _TreeNode()

            node = node.children[ch]

        node.is_token = True

    def get_longest_match(self, text: str) -> str:
        longest_match = ""
        path = []

        node = self.root

        for ch in text:
            if ch not in node.children:
                break

            node = node.children[ch]
            path.append(ch)

            if node.is_token:
                longest_match = "".join(path)

        return longest_match


class SpecialTokenMixin:
    @property
    def bos_token(self) -> str:
        return self._special_tokens["bos_token"]

    @property
    def bos_token_id(self) -> int:
        return self._vocab[self.bos_token]

    @property
    def eos_token(self) -> str:
        return self._special_tokens["eos_token"]

    @property
    def eos_token_id(self) -> int:
        return self._vocab[self.eos_token]

    @property
    def pad_token(self) -> str:
        return self._special_tokens["pad_token"]

    @property
    def pad_token_id(self) -> int:
        return self._vocab[self.pad_token]

    @property
    def unk_token(self) -> str:
        return self._special_tokens["unk_token"]

    @property
    def unk_token_id(self) -> int:
        return self._vocab[self.unk_token]

    @property
    def mask_token(self) -> str:
        return self._special_tokens["mask_token"]

    @property
    def mask_token_id(self) -> int:
        return self._vocab[self.mask_token]


class Tokenizer(SpecialTokenMixin):
    def __init__(
        self, vocab: Dict[str, int], special_tokens: Dict[str, str], chat_template: str
    ):
        self._max_token_len = max(map(lambda key: len(key), vocab.keys()))
        self._vocab_tree = _VocabTree()
        for token in vocab.keys():
            self._vocab_tree.add_token(token)

        self._vocab = vocab
        self._rev_vocab = {val: key for key, val in vocab.items()}
        self._special_tokens = special_tokens
        self._chat_template = Template(source=chat_template)

        for special_token_name, special_token in special_tokens.items():
            assert special_token in vocab, (
                f"Special token {special_token_name} is defined but not in vocab"
            )

    @property
    def vocab_size(self) -> int:
        return max(self._vocab.values()) + 1

    @property
    def max_token_len(self) -> int:
        return self._max_token_len

    @property
    def chat_template(self) -> Template:
        return self._chat_template

    def apply_chat_template(self, messages: List[Message]) -> str:
        return self._chat_template.render(
            messages=messages,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
        )

    def __call__(self, text: str, add_special_tokens: bool = False) -> List[int]:
        token_ids: List[int] = []

        start: int = 0
        end: int = self.max_token_len

        # get the biggest piece possible that could match
        window: str = text[start:end]

        while len(window) > 0:
            longest_match = self._vocab_tree.get_longest_match(window)

            token_id: int = self._vocab.get(longest_match) or self.unk_token_id
            token_ids.append(token_id)

            start = start + max(len(longest_match), 1)
            end = start + self.max_token_len

            # get the biggest piece possible that could match
            window: str = text[start:end]

        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
        return token_ids

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return self.__call__(text, add_special_tokens=add_special_tokens)

    def decode(self, token_ids: List[int]) -> str:
        return "".join(
            self._rev_vocab.get(token_idx) or self.unk_token for token_idx in token_ids
        )

    @classmethod
    def from_json_file(cls, tokenizer_file_path: str):
        with open(tokenizer_file_path, "r") as foo:
            tokenizer_config = json.load(foo)

        assert "special_tokens" in tokenizer_config
        assert "chat_template" in tokenizer_config
        assert "vocab" in tokenizer_config

        return cls(
            tokenizer_config["vocab"],
            tokenizer_config["special_tokens"],
            tokenizer_config["chat_template"],
        )


## ------------------- Modelling ------------------- ##


@dataclass
class ModelConfig:
    hidden_size: int
    vocab_size: int
    num_layers: int
    max_seq_len: int

    rope_theta: int

    num_q_heads: int
    num_kv_heads: int
    attention_bias: bool

    padding_idx: int
    rms_norm_eps: float

    mlp_inter_hidden_size: int
    mlp_bias: bool


class MLP(nn.Module):
    # ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L162

    def __init__(self, hidden_size: int, inter_hidden_size: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, inter_hidden_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, inter_hidden_size, bias=bias)
        self.down_proj = nn.Linear(inter_hidden_size, hidden_size, bias=bias)
        self.act_fn = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class RoPEMultiHeadAttentionWithGQA(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert num_q_heads > num_kv_heads
        assert num_q_heads % num_kv_heads == 0

        self.head_dim = hidden_size // num_q_heads
        self.scale = self.head_dim**-0.5
        self.num_kv_groups = num_q_heads // num_kv_heads
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(hidden_size, num_q_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_q_heads * self.head_dim, hidden_size, bias=bias)

    def forward(
        self,
        x: Tensor,
        pos_embeddings: Tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """_summary_

        Args:
            x (Tensor): B x S x d
            pos_embeddings (Tuple[Tensor, Tensor]): cos_theta with (S_max x d) shape and sin_theta with (S_max x d) shape
            attention_mask (Optional[Tensor], optional): B x 1 x S x S mask if exists. Defaults to None.

        Returns:
            Tensor: B x S x d
        """
        batch_size, seq_len, _ = x.shape

        q: Tensor = (
            self.q_proj(x)
            .reshape(
                batch_size,
                seq_len,
                self.num_q_heads,
                self.head_dim,
            )
            .permute(0, 2, 1, 3)
        )
        # q: B x nh x S x d_h
        k: Tensor = (
            self.k_proj(x)
            .reshape(
                batch_size,
                seq_len,
                self.num_kv_heads,
                self.head_dim,
            )
            .permute(0, 2, 1, 3)
        )
        # k: B x nkv x S x d_h
        v: Tensor = (
            self.v_proj(x)
            .reshape(
                batch_size,
                seq_len,
                self.num_kv_heads,
                self.head_dim,
            )
            .permute(0, 2, 1, 3)
        )
        # v: B x nkv x S x d_h

        cos_theta, sin_theta = pos_embeddings
        q = apply_rope_embed(
            q.flatten(start_dim=0, end_dim=1),
            cos_theta,
            sin_theta,
        ).unflatten(dim=0, sizes=(batch_size, self.num_q_heads))
        # q: B x nh x S x d_h
        k = apply_rope_embed(
            k.flatten(start_dim=0, end_dim=1),
            cos_theta,
            sin_theta,
        ).unflatten(dim=0, sizes=(batch_size, self.num_kv_heads))
        # k: B x nkv x S x d_h

        out: Tensor = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout,
            is_causal=attention_mask is None,
            scale=self.scale,
            enable_gqa=True,
        )
        # out: B x nh x S x d_h

        out: Tensor = self.o_proj(
            out.permute(0, 2, 1, 3).flatten(start_dim=2, end_dim=3)
        )
        # out: B x S x d

        return out


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = RoPEMultiHeadAttentionWithGQA(
            config.hidden_size,
            config.num_q_heads,
            config.num_kv_heads,
            bias=config.attention_bias,
        )

        self.mlp = MLP(
            config.hidden_size, config.mlp_inter_hidden_size, bias=config.mlp_bias
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        x: Tensor,
        pos_embeddings: Tuple[Tensor, Tensor],
        attention_mask: BoolTensor = None,
    ) -> Tensor:
        res = x
        x = self.input_layernorm(x)

        # self attention
        x = self.self_attn(x, pos_embeddings, attention_mask=attention_mask)

        x = res + x

        # fully connected
        res = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = res + x

        return x


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, 1, hidden_size), requires_grad=True)
        self.register_buffer("eps", torch.tensor(eps), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Applies RMS Normalisation

        Args:
            x (Tensor): x with shape of B x S x d

        Returns:
            Tensor: outputs with shape of B x S x d
        """
        out = x.pow(2).mean(dim=2, keepdim=True) + self.eps
        out = x / torch.sqrt(out)
        return self.weight * out


class RotaryEmbedding(nn.Module):
    def __init__(self, hidden_size: int, max_seq_len: int, theta: int = 10000):
        super().__init__()
        assert hidden_size % 2 == 0, "hidden size must be even."

        d_half = hidden_size // 2

        theta: Tensor = theta ** (
            -1 * torch.arange(d_half, dtype=torch.float32).unsqueeze(0) / d_half
        )
        # theta: 1 x d_half
        m: Tensor = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(0)
        # m: 1 x S

        self.register_buffer(
            "cos_theta",
            torch.cos(m.T @ theta).repeat(
                1, 2
            ),  # Since we do rotate half in a different way
            persistent=False,
        )
        # cos_theta: S x d

        self.register_buffer(
            "sin_theta",
            torch.sin(m.T @ theta).repeat(
                1, 2
            ),  # Since we do rotate half in a different way
            persistent=False,
        )
        # sin_theta: S x d


def rotate_half(x: Tensor) -> Tensor:
    # ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L128
    # See below to understand why it's different from original llama
    # https://github.com/huggingface/transformers/issues/25199
    _, _, hidden_dim = x.shape

    x1 = x[:, :, : hidden_dim // 2]
    x2 = x[:, :, hidden_dim // 2 :]

    return torch.cat((-x2, x1), dim=2)


def apply_rope_embed(x: Tensor, cos_theta: Tensor, sin_theta: Tensor) -> Tensor:
    """Applies rope embedding with pre-computed cos_theta and sin_theta

    Args:
        x (Tensor): B x S x d
        cos_theta (Tensor): S_max x d
        sin_theta (Tensor): S_max x d

    Returns:
        Tensor: B x S x d
    """
    seq_len = x.shape[1]

    return x * cos_theta[:seq_len, :] + rotate_half(x) * sin_theta[:seq_len, :]


class SmolLM2(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.padding_idx
        )
        self.rotary_emb = RotaryEmbedding(
            config.hidden_size // config.num_q_heads,
            config.max_seq_len,
            theta=config.rope_theta,
        )
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: LongTensor,
        attention_mask: BoolTensor = None,
    ) -> Tensor:
        x: Tensor = self.embed_tokens(input_ids)
        # x: B x S x d

        pos_embeddings: Tuple[Tensor, Tensor] = (
            self.rotary_emb.cos_theta,
            self.rotary_emb.sin_theta,
        )

        for decoder_layer in self.layers:
            x = decoder_layer(x, pos_embeddings, attention_mask=attention_mask)

        x = self.norm(x)

        return self.lm_head(x)

    @classmethod
    def from_checkpoint(cls, ckpt_file_path: str, device: str = "cpu"):
        ckpt = torch.load(ckpt_file_path, map_location=device)
        config = ModelConfig(**ckpt["config"])
        model = cls(config)
        model.load_state_dict(ckpt["weights"])
        return model


## ------------------- Optimization ------------------- ##


def training_loop(
    model: SmolLM2, dl, criterion, optimizer, device, verbosity: int = 10
) -> float:
    model.train()

    total_loss = []
    accumulated_loss = []
    for batch, targets, attn_mask in tqdm(dl):
        optimizer.zero_grad()
        logits = model(batch.to(device), attention_mask=attn_mask.to(device))
        loss = criterion(
            logits.flatten(start_dim=0, end_dim=1),
            targets.flatten(start_dim=0, end_dim=1).to(device),
        )
        accumulated_loss.append(loss.item())
        total_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        if len(accumulated_loss) >= verbosity:
            print(
                "Training loss -> {:.3f}".format(
                    sum(accumulated_loss) / len(accumulated_loss)
                )
            )
            accumulated_loss = []

    return sum(total_loss) / len(total_loss)


def validation_loop(model, dl, criterion, device) -> float:
    return _non_training_loop(model, dl, criterion, device)


def test_loop(model, dl, criterion, device) -> float:
    return _non_training_loop(model, dl, criterion, device)


@torch.no_grad()
def _non_training_loop(model, dl, criterion, device) -> float:
    model.eval()
    total_loss = []
    for batch, targets, attn_mask in tqdm(dl):
        logits = model(batch.to(device), attention_mask=attn_mask.to(device))
        loss = criterion(
            logits.flatten(start_dim=0, end_dim=1),
            targets.flatten(start_dim=0, end_dim=1).to(device),
        )
        total_loss.append(loss.item())
    return sum(total_loss) / len(total_loss)


@torch.no_grad()
def sample(
    model: SmolLM2,
    tokenizer: Tokenizer,
    messages: Tuple[Message, Message],
    device: str,
    max_tokens: int = 50,
) -> Generator[str, None, None]:
    model.eval()

    input_raw = tokenizer.apply_chat_template(messages)
    print("prompt;\n", input_raw)
    input_tokens: List[int] = tokenizer(input_raw)
    input_ids = torch.tensor(input_tokens, dtype=torch.long, device=device).unsqueeze(0)

    # input_ids: 1 x S
    generated_token_id = None
    generated_token_counter = 0
    while generated_token_id != tokenizer.eos_token_id:
        seq_len = input_ids.shape[1]
        attn_mask = torch.ones(
            1, seq_len, seq_len, dtype=torch.bool, device=device
        ).tril(diagonal=0)

        logits = model(input_ids, attn_mask)
        # logits: 1 x S x C

        probs = F.softmax(logits[:, -1, :], dim=1)
        # probs: 1 x C

        generated_token_id = torch.multinomial(probs, num_samples=1)
        # generated_token_id: 1 x 1

        input_ids = torch.cat([input_ids, generated_token_id], dim=1)
        # input_ids: 1 x (S + 1)

        generated_token_id = generated_token_id.flatten().item()

        yield tokenizer.decode([generated_token_id])

        generated_token_counter += 1

        if generated_token_counter >= max_tokens:
            break


def main(args):
    model_name: str = args.model_name
    data_path: str = args.data_path
    model_path: str = args.model_path
    tokenizer = Tokenizer.from_json_file(
        os.path.join(model_path, f"{model_name}-tokenizer.json")
    )
    train_ds = InstructDataset.from_jsonl(
        os.path.join(data_path, "train.jsonl"), tokenizer
    )
    val_ds = InstructDataset.from_jsonl(os.path.join(data_path, "val.jsonl"), tokenizer)
    test_ds = InstructDataset.from_jsonl(
        os.path.join(data_path, "test.jsonl"), tokenizer
    )
    device: str = args.device
    dtype = torch.bfloat16

    model = SmolLM2.from_checkpoint(os.path.join(model_path, f"{model_name}-it.ckpt"))

    if ((num_available_gpus := torch.cuda.device_count()) > 1) and device.startswith(
        "cuda"
    ):
        print(f"{num_available_gpus} GPUs found, switching to nn.DataParallel")
        model = nn.DataParallel(model)

    model.to(device, dtype)

    learning_rate: float = args.learning_rate
    epoch: int = args.epoch
    batch_size: int = args.batch_size
    ignore_index: int = args.ignore_index

    collate_fn = partial(
        dynamic_collate_fn,
        pad_token_id=tokenizer.pad_token_id,
        ignore_index=ignore_index,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=args.num_process,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_process,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_process,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    best_val_loss = math.inf

    for i in range(epoch):
        print("running random 5 samples from validation set to sanity check")
        print("---" * 20)
        # get random samples
        ids = random.sample(list(range(len(val_ds))), k=5)
        for idx in ids:
            system_message, user_message, _ = deepcopy(val_ds._data[idx])
            print("Response;")
            for token in sample(
                model, tokenizer, (system_message, user_message), device, max_tokens=100
            ):
                print(token, flush=True, end="")
            print()
        print("---" * 20)
        train_loss = training_loop(model, train_dl, criterion, optimizer, device)
        print(f"[{i + 1}/{epoch}] Epoch | Training Loss : {train_loss:.3f}")
        val_loss = validation_loop(model, val_dl, criterion, device)
        print(f"[{i + 1}/{epoch}] Epoch | Validation Loss : {val_loss:.3f}")
        if val_loss < best_val_loss:
            print(
                f"Found a better model {best_val_loss:.2f} -> {val_loss:.2f}, saving..."
            )
            best_val_loss = val_loss
            if num_available_gpus > 1:
                config = asdict(model.module.config)
                weights = model.module.state_dict()
            else:
                config = asdict(model.config)
                weights = model.state_dict()

            torch.save(
                {"config": config, "weights": weights},
                os.path.join(model_path, f"{model_name}-best-full-sft.ckpt"),
            )

    test_loss = test_loop(model, test_dl, criterion, device)
    print(f"Test Loss : {test_loss:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-name",
        "-m",
        type=str,
        choices=["smollm2-135m", "smollm2-360m"],
        required=True,
    )
    ap.add_argument("--data-path", "-dp", type=str, default="data")
    ap.add_argument("--model-path", "-mp", type=str, default="artifacts")
    ap.add_argument("--device", "-d", type=str, default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--num-process", "-ns", type=int, default=16)
    ap.add_argument("--learning-rate", "-lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", "-bs", type=int, default=32)
    ap.add_argument("--epoch", "-e", type=int, default=10)
    ap.add_argument("--ignore-index", "-ig", type=int, default=-100)

    main(ap.parse_args())
