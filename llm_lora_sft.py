from typing import Literal, List, Dict, Tuple, Optional, Generator, Iterator
import argparse
import os
import json
from jinja2 import Template
from dataclasses import dataclass, field
from copy import deepcopy
from tqdm import tqdm
from functools import partial
import math
import random
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    and_masks,
    BlockMask,
)
from torch import Tensor, LongTensor, BoolTensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

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
    batch_padding_mask = []

    for token_ids in all_token_ids:
        token_ids = torch.tensor(token_ids, dtype=torch.long)

        input_ids = token_ids[:-1]
        targets = token_ids[1:]

        seq_len = input_ids.shape[0]

        num_pads = max_seq_len - seq_len

        input_ids = F.pad(input_ids, (0, num_pads), value=pad_token_id)
        targets = F.pad(targets, (0, num_pads), value=ignore_index)
        padding_mask = torch.zeros(max_seq_len, dtype=torch.bool)
        padding_mask[seq_len:] = True

        batch_input_ids.append(input_ids)
        batch_targets.append(targets)
        batch_padding_mask.append(padding_mask)

    batch_input_ids = torch.stack(batch_input_ids)
    batch_targets = torch.stack(batch_targets)
    batch_padding_mask = torch.stack(batch_padding_mask)

    return batch_input_ids, batch_targets, batch_padding_mask


## ------------------- Multi GPU ------------------- ##


@dataclass
class MultiGPUConfig:
    rank: int = field(default=lambda: int(os.environ["RANK"]))
    local_rank: int = field(default=lambda: int(os.environ["LOCAL_RANK"]))
    world_size: int = field(default=lambda: int(os.environ["WORLD_SIZE"]))


def ddp_setup(rank: int, world_size: int):
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)


def ddp_destroy():
    torch.distributed.destroy_process_group()


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

    rope_global_theta: int
    rope_local_theta: int

    head_dim: int
    num_q_heads: int
    num_kv_heads: int
    attention_bias: bool
    is_sliding_attention: List[bool]
    window_size: int

    padding_idx: int
    rms_norm_eps: float

    mlp_inter_hidden_size: int
    mlp_bias: bool


class MLP(nn.Module):
    # ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modeling_gemma3.py#L147

    def __init__(self, hidden_size: int, inter_hidden_size: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, inter_hidden_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, inter_hidden_size, bias=bias)
        self.down_proj = nn.Linear(inter_hidden_size, hidden_size, bias=bias)
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x: Tensor) -> Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class RoPEMultiHeadAttentionWithGQA(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        bias: bool = False,
        dropout: float = 0.0,
        eps: float = 1e-8,
        window_size: int | None = None,
    ):
        super().__init__()

        assert num_q_heads > num_kv_heads
        assert num_q_heads % num_kv_heads == 0

        self.head_dim = head_dim
        self.scale = self.head_dim**-0.5
        self.num_kv_groups = num_q_heads // num_kv_heads
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.dropout = dropout
        self.is_sliding = bool(window_size)
        self.window_size = window_size

        self.q_proj = nn.Linear(hidden_size, num_q_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_q_heads * self.head_dim, hidden_size, bias=bias)

        self.q_norm = RMSNorm(self.head_dim, eps=eps)
        self.k_norm = RMSNorm(self.head_dim, eps=eps)

    @staticmethod
    def _sliding_window_mask_mod(window_size: int):
        # `True` means allow attention
        def _sliding_window(b, h, q_idx, kv_idx):
            return (q_idx - kv_idx) <= window_size

        return _sliding_window

    @staticmethod
    def _causal_mask_mod():
        # `True` means allow attention
        def _causal(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        return _causal

    @staticmethod
    def _padding_mask_mod(padding_mask: BoolTensor):
        # `True` means allow attention
        def _padding(b, h, q_idx, kv_idx):
            return ~(padding_mask[b, q_idx] | padding_mask[b, kv_idx])

        return _padding

    def forward(
        self,
        x: Tensor,
        pos_embeddings: Tuple[Tensor, Tensor],
        padding_mask: Optional[BoolTensor] = None,
    ) -> Tensor:
        """_summary_

        Args:
            x (Tensor): B x S x d
            pos_embeddings (Tuple[Tensor, Tensor]): cos_theta with (S_max x d) shape and sin_theta with (S_max x d) shape
            padding_mask (Optional[Tensor], optional): B x S mask if exists, True means it is padded. Defaults to None.

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

        q: Tensor = self.q_norm(q)
        k: Tensor = self.k_norm(k)

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

        masks = []

        # add causal mask
        masks.append(self._causal_mask_mod())

        # add sliding masking if needed
        if self.is_sliding:
            masks.append(self._sliding_window_mask_mod(self.window_size))

        # add padding masking if exists
        padding_exists = padding_mask is not None
        if padding_exists:
            masks.append(self._padding_mask_mod(padding_mask))

        attn_mask_mod = and_masks(*masks)

        # construct the block mask
        block_mask: BlockMask = create_block_mask(
            mask_mod=attn_mask_mod,
            B=batch_size,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=q.device,
        )

        out: Tensor = flex_attention(
            q,
            k,
            v,
            block_mask=block_mask,
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
    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        mlp_inter_hidden_size: int,
        mlp_bias: bool = False,
        attention_bias: bool = False,
        window_size: int | None = None,
        rms_norm_eps: float = 1e-8,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.self_attn = RoPEMultiHeadAttentionWithGQA(
            hidden_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            bias=attention_bias,
            eps=rms_norm_eps,
            window_size=window_size,
        )

        self.mlp = MLP(hidden_size, mlp_inter_hidden_size, bias=mlp_bias)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(self.hidden_size, eps=rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(self.hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        x: Tensor,
        pos_embeddings: Tuple[Tensor, Tensor],
        padding_mask: BoolTensor = None,
    ) -> Tensor:
        res = x
        x = self.input_layernorm(x)

        # self attention
        x = self.self_attn(x, pos_embeddings, padding_mask=padding_mask)

        x = self.post_attention_layernorm(x)
        x = res + x

        # fully connected
        res = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        x = res + x

        return x


class RMSNorm(nn.Module):
    # ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modeling_gemma3.py#L152

    def __init__(self, hidden_size: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
        self.eps = eps

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        """Applies RMS Normalisation

        Args:
            x (Tensor): x with shape of B x ... x S x d

        Returns:
            Tensor: outputs with shape of B x ... x S x d
        """
        # type casting is required, see here: https://github.com/huggingface/transformers/pull/29402
        # TODO add type casting to float if needed
        output = self._norm(x)
        output = output * (1.0 + self.weight)
        return output

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


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
    # ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modeling_gemma3.py#L217
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


class ScaledEmbedding(nn.Embedding):
    # ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modeling_gemma3.py#L134

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: float = 1.0,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale


class Gemma3(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = ScaledEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.padding_idx,
            embed_scale=config.hidden_size**0.5,
        )
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    config.hidden_size,
                    config.num_q_heads,
                    config.num_kv_heads,
                    config.head_dim,
                    config.mlp_inter_hidden_size,
                    mlp_bias=config.mlp_bias,
                    attention_bias=config.attention_bias,
                    window_size=config.window_size
                    if config.is_sliding_attention[layer_idx]
                    else None,
                    rms_norm_eps=config.rms_norm_eps,
                )
                for layer_idx in range(config.num_layers)
            ]
        )
        self.rotary_emb_gobal = RotaryEmbedding(
            config.head_dim,
            config.max_seq_len,
            theta=config.rope_global_theta,
        )
        self.rotary_emb_local = RotaryEmbedding(
            config.head_dim,
            config.max_seq_len,
            theta=config.rope_local_theta,
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: LongTensor,
        padding_mask: BoolTensor = None,
    ) -> Tensor:
        x: Tensor = self.embed_tokens(input_ids)
        # x: B x S x d

        global_pos_embeddings: Tuple[Tensor, Tensor] = (
            self.rotary_emb_gobal.cos_theta,
            self.rotary_emb_gobal.sin_theta,
        )

        local_pos_embeddings: Tuple[Tensor, Tensor] = (
            self.rotary_emb_local.cos_theta,
            self.rotary_emb_local.sin_theta,
        )

        for i, decoder_layer in enumerate(self.layers):
            if decoder_layer.self_attn.is_sliding:
                pos_embeddings = local_pos_embeddings
            else:
                pos_embeddings = global_pos_embeddings

            x = decoder_layer(
                x,
                pos_embeddings,
                padding_mask=padding_mask,
            )

        x = self.norm(x)

        return self.lm_head(x)

    @classmethod
    def from_checkpoint(cls, ckpt_file_path: str, device: str = "cpu"):
        ckpt = torch.load(ckpt_file_path, map_location=device)
        config = ModelConfig(**ckpt["config"])
        model = cls(config)
        model.load_state_dict(ckpt["weights"])
        return model


class LoRALinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, rank: int, alpha: float = 1.0
    ):
        super().__init__()
        assert rank < min(in_features, out_features), (
            "rank must be way smaller than original features"
        )
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.A = nn.Parameter(torch.randn(size=(in_features, rank)), requires_grad=True)
        self.B = nn.Parameter(
            torch.zeros(size=(rank, out_features)), requires_grad=True
        )
        self.register_buffer("scaler", torch.tensor(alpha / rank), persistent=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.A.data.normal_(mean=0, std=1)
        self.B.data.fill_(0)

    def forward(self, x: Tensor) -> Tensor:
        return ((x @ self.A) @ self.B) * self.scaler

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, scaler={self.scaler}"


class CombinedLinear(nn.Module):
    def __init__(self, full_rank_linear: nn.Linear, low_rank_linear: LoRALinear):
        super().__init__()
        self.full_rank_linear = full_rank_linear
        self.low_rank_linear = low_rank_linear

    def forward(self, x: Tensor) -> Tensor:
        return self.full_rank_linear(x) + self.low_rank_linear(x)


class LoraAdaptor:
    def __init__(self, model: Gemma3, rank: int, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.model = model

    def register_layers(self, layer_types=None):
        assert layer_types is not None

        for name, module in self.model.named_modules():
            if not isinstance(module, layer_types):
                continue

            linear_layers = filter(
                lambda sub_module: isinstance(sub_module[1], nn.Linear),
                module.named_modules(),
            )

            for subname, linear_layer in linear_layers:
                # define the lora layer
                lora_layer = LoRALinear(
                    linear_layer.in_features,
                    linear_layer.out_features,
                    self.rank,
                    alpha=self.alpha,
                )
                setattr(module, subname, CombinedLinear(linear_layer, lora_layer))

    def freeze_model(self):
        # freeze all
        self.model.requires_grad_(False)

        # unfreeze lora weights
        for name, param in self.model.named_parameters():
            if "low_rank_linear" in name:
                param.requires_grad_(True)

    def parameters(self) -> Iterator[nn.Parameter]:
        for name, param in self.model.named_parameters():
            if "low_rank_linear" in name:
                yield param

    def state_dict(self) -> OrderedDict:
        model = self.model.module if hasattr(self.model, "module") else self.model
        state_dict = OrderedDict()
        for key, value in model.state_dict().items():
            if "low_rank_linear" in key:
                state_dict[key] = value
        return state_dict


## ------------------- Optimization ------------------- ##


def training_loop(
    model: Gemma3, dl, criterion, optimizer, device, verbosity: int = 10
) -> float:
    total_loss = []
    accumulated_loss = []
    for batch, targets, padding_mask in tqdm(dl):
        optimizer.zero_grad()
        logits = model(batch.to(device), padding_mask=padding_mask.to(device))
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
    total_loss = []
    for batch, targets, padding_mask in tqdm(dl):
        logits = model(batch.to(device), padding_mask=padding_mask.to(device))
        loss = criterion(
            logits.flatten(start_dim=0, end_dim=1),
            targets.flatten(start_dim=0, end_dim=1).to(device),
        )
        total_loss.append(loss.item())
    return sum(total_loss) / len(total_loss)


@torch.no_grad()
def sample(
    model: Gemma3,
    tokenizer: Tokenizer,
    messages: Tuple[Message, Message],
    device: str,
    max_tokens: int = 50,
) -> Generator[str, None, None]:
    input_raw = tokenizer.apply_chat_template(messages)
    print("prompt;\n", input_raw)
    input_tokens: List[int] = tokenizer(input_raw)
    input_ids = torch.tensor(input_tokens, dtype=torch.long, device=device).unsqueeze(0)

    # input_ids: 1 x S
    generated_token_id = None
    generated_token_counter = 0
    while generated_token_id != tokenizer.eos_token_id:
        logits = model(input_ids)
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
    num_available_gpus: int = torch.cuda.device_count()
    is_multi_gpu_training: bool = num_available_gpus > 1 and device.startswith("cuda")
    multi_gpu_config: MultiGPUConfig = None

    model = Gemma3.from_checkpoint(os.path.join(model_path, f"{model_name}-it.ckpt"))
    model.eval()

    if is_multi_gpu_training:
        print(f"{num_available_gpus} GPUs found, switching to DDP")
        multi_gpu_config = MultiGPUConfig()
        torch.cuda.set_device(multi_gpu_config.local_rank)
        ddp_setup(multi_gpu_config.rank, multi_gpu_config.world_size)
        device: str = f"cuda:{multi_gpu_config.local_rank}"
        model.to(device)
        model = DDP(model, device_ids=[multi_gpu_config.local_rank])

    is_master = (not is_multi_gpu_training) or (torch.distributed.get_rank() == 0)

    lora_adaptor = LoraAdaptor(model, args.rank, alpha=args.alpha)

    lora_adaptor.register_layers(layer_types=(RoPEMultiHeadAttentionWithGQA,))

    model.to(device, dtype)

    # freeze the main model
    lora_adaptor.freeze_model()

    learning_rate: float = args.learning_rate
    epoch: int = args.epoch
    batch_size: int = args.batch_size
    ignore_index: int = args.ignore_index

    collate_fn = partial(
        dynamic_collate_fn,
        pad_token_id=tokenizer.pad_token_id,
        ignore_index=ignore_index,
    )

    train_sampler = None
    val_sampler = None
    test_sampler = None

    if is_multi_gpu_training:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=multi_gpu_config.world_size,
            rank=multi_gpu_config.rank,
        )
        val_sampler = DistributedSampler(
            val_ds,
            num_replicas=multi_gpu_config.world_size,
            rank=multi_gpu_config.rank,
        )
        test_sampler = DistributedSampler(
            test_ds,
            num_replicas=multi_gpu_config.world_size,
            rank=multi_gpu_config.rank,
        )

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=train_sampler is None,
        num_workers=args.num_process,
        sampler=train_sampler,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_process,
        sampler=val_sampler,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_process,
        sampler=test_sampler,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    optimizer = torch.optim.AdamW(
        lora_adaptor.parameters(),
        lr=learning_rate,
    )

    best_val_loss = math.inf
    num_infer_samples: int = 1

    for i in range(epoch):
        if is_master:
            print(
                f"running random {num_infer_samples} samples from validation set to sanity check"
            )
            print("---" * 20)
            # get random samples
            ids = random.sample(list(range(len(val_ds))), k=num_infer_samples)
            for idx in ids:
                system_message, user_message, _ = deepcopy(val_ds._data[idx])
                print("Response;")
                for token in sample(
                    model,
                    tokenizer,
                    (system_message, user_message),
                    device,
                    max_tokens=100,
                ):
                    print(token, flush=True, end="")
                print()
            print("---" * 20)
        train_loss = training_loop(model, train_dl, criterion, optimizer, device)
        if is_master:
            print(f"[{i + 1}/{epoch}] Epoch | Training Loss : {train_loss:.3f}")
        val_loss = validation_loop(model, val_dl, criterion, device)
        if is_master:
            print(f"[{i + 1}/{epoch}] Epoch | Validation Loss : {val_loss:.3f}")
        if val_loss < best_val_loss and is_master:
            print(
                f"Found a better model {best_val_loss:.2f} -> {val_loss:.2f}, saving..."
            )
            best_val_loss = val_loss

            torch.save(
                lora_adaptor.state_dict(),
                os.path.join(model_path, f"{model_name}-best-lora-sft.pt"),
            )

    test_loss = test_loop(model, test_dl, criterion, device)
    if is_master:
        print(f"Test Loss : {test_loss:.3f}")

    if is_multi_gpu_training:
        ddp_destroy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-name",
        "-m",
        type=str,
        choices=["gemma3-1b"],
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
    ap.add_argument("--rank", "-r", type=int, required=True)
    ap.add_argument("--alpha", "-a", type=float, default=1.0)

    main(ap.parse_args())
