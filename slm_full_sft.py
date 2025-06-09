from typing import Literal, List, Dict, Tuple, Optional, Generator
import argparse
import os
import json
from jinja2 import Template
from dataclasses import dataclass, asdict, field
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

            # Sanity check
            # TODO think about tool calling mb
            assert len(messages) >= 3, (
                "There should be at least 3 messages which belongs to `system`, `user` and `assistant`"
            )
            assert messages[0].role == "system", (
                "First message in the dataset must be `system` message"
            )
            assert len(messages[1:]) % 2 == 0, (
                "Number of messages after system message must be divisible by 2"
            )
            assert (
                all(
                    [
                        message.role == "user"  # first message must be coming from user
                        if i % 2 == 0
                        else message.role
                        == "assistant"  # last message must be coming from assistant
                        for i, message in enumerate(messages[1:])
                    ]
                )
                == True
            ), (
                "Sanity check failed for multi-turn messages, pattern must be system -> user -> assistant -> user -> assistant ..."
            )
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
    rank: int = field(default_factory=lambda: int(os.environ["RANK"]))
    local_rank: int = field(default_factory=lambda: int(os.environ["LOCAL_RANK"]))
    world_size: int = field(default_factory=lambda: int(os.environ["WORLD_SIZE"]))


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

        self.reset_cache()

    def reset_cache(self):
        """Needs to be called manually when causal inference is done."""
        self.k_cache = torch.empty((1, self.num_kv_heads, 0, self.head_dim))
        self.v_cache = torch.empty((1, self.num_kv_heads, 0, self.head_dim))

    @staticmethod
    def get_attn_mask(
        batch_size: int,
        seq_len: int,
        device: str,
        pos_offset: int,
        padding_mask: Optional[BoolTensor] = None,
    ) -> BoolTensor:
        """Computed attention mask, True means allow attention

        Args:
            batch_size (int): batch size of the input
            seq_len (int): current sequence length
            device (str): device where input lives
            pos_offset (int): prev cache sequence length
            padding_mask (Optional[BoolTensor], optional): B x S mask if exists, True means it is padded. Defaults to None.

        Returns:
            BoolTensor: Final attention mask with shape of B x 1 x S x S_full
        """

        attn_mask: BoolTensor = torch.ones(
            batch_size,
            seq_len,
            seq_len + pos_offset,
            dtype=torch.bool,
            device=device,
        ).tril(diagonal=pos_offset)
        # attn_mask: B x S x S_full

        if padding_mask is not None:
            assert pos_offset == 0, (
                "when padding mask exists, assuming KV cache is not being used"
            )
            attn_mask.masked_fill_(padding_mask.unsqueeze(1), False)
            # attn_mask: B x S x S

        return attn_mask.unsqueeze(1)

    def forward(
        self,
        x: Tensor,
        pos_embeddings: Tuple[Tensor, Tensor],
        padding_mask: Optional[BoolTensor] = None,
        use_cache: bool = False,
    ) -> Tensor:
        """Causal MHA with RoPE and GQA

        Args:
            x (Tensor): B x S x d
            pos_embeddings (Tuple[Tensor, Tensor]): cos_theta with (S_max x d) shape and sin_theta with (S_max x d) shape
            padding_mask (Optional[BoolTensor], optional): B x S mask if exists, True means it is padded. Defaults to None.
            use_cache bool: whether to use KV cache or not

        Returns:
            Tensor: B x S x d
        """
        batch_size, seq_len, _ = x.shape

        pos_offset: int = self.k_cache.shape[2]

        assert (not use_cache) or (batch_size == 1), (
            "If cache is enabled, only batch size 1 supported"
        )

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

        if use_cache:
            # register it to cache
            self.k_cache = torch.cat([self.k_cache.to(k.device, k.dtype), k], dim=2)
            # k_cache: B x nkv x S_full x d_h

            # register it to cache
            self.v_cache = torch.cat([self.v_cache.to(v.device, v.dtype), v], dim=2)
            # v_cache: B x nkv x S_full x d_h

            k = self.k_cache.clone()
            v = self.v_cache.clone()

        cos_theta, sin_theta = pos_embeddings
        q = apply_rope_embed(
            q.flatten(start_dim=0, end_dim=1),
            cos_theta,
            sin_theta,
            pos_offset=pos_offset,
        ).unflatten(dim=0, sizes=(batch_size, self.num_q_heads))
        # q: B x nh x S x d_h
        k = apply_rope_embed(
            k.flatten(start_dim=0, end_dim=1),
            cos_theta,
            sin_theta,
        ).unflatten(dim=0, sizes=(batch_size, self.num_kv_heads))
        # k: B x nkv x S x d_h

        attn_mask: BoolTensor = self.get_attn_mask(
            batch_size,
            seq_len,
            x.device,
            pos_offset,
            padding_mask=padding_mask,
        )
        # attn_mask: B x 1 x S x S_full

        out: Tensor = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
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
            dropout=0.3,
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
        padding_mask: BoolTensor = None,
        use_cache: bool = False,
    ) -> Tensor:
        res = x
        x = self.input_layernorm(x)

        # self attention
        x = self.self_attn(
            x, pos_embeddings, padding_mask=padding_mask, use_cache=use_cache
        )

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


def apply_rope_embed(
    x: Tensor, cos_theta: Tensor, sin_theta: Tensor, pos_offset: int = 0
) -> Tensor:
    """Applies rope embedding with pre-computed cos_theta and sin_theta

    Args:
        x (Tensor): B x S x d
        cos_theta (Tensor): S_max x d
        sin_theta (Tensor): S_max x d
        pos_offset (int, optional): Offset of the position due to KV cache. Defaults to 0.

    Returns:
        Tensor: B x S x d
    """
    seq_len = x.shape[1]

    return (
        x * cos_theta[pos_offset : pos_offset + seq_len, :]
        + rotate_half(x) * sin_theta[pos_offset : pos_offset + seq_len, :]
    )


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

    def decoder(
        self,
        x: Tensor,
        padding_mask: BoolTensor = None,
        use_cache: bool = False,
    ) -> Tensor:
        pos_embeddings: Tuple[Tensor, Tensor] = (
            self.rotary_emb.cos_theta,
            self.rotary_emb.sin_theta,
        )

        for decoder_layer in self.layers:
            x = decoder_layer(
                x, pos_embeddings, padding_mask=padding_mask, use_cache=use_cache
            )

        x = self.norm(x)

        return x

    def forward(
        self,
        input_ids: LongTensor,
        padding_mask: BoolTensor = None,
        use_cache: bool = False,
    ) -> Tensor:
        x: Tensor = self.embed_tokens(input_ids)
        # x: B x S x d

        x = self.decoder(x, padding_mask=padding_mask, use_cache=use_cache)

        return self.lm_head(x)

    @torch.no_grad()
    def sample(
        self,
        input_ids: LongTensor,
        stop_token: int,
        max_tokens: int = 200,
    ) -> Generator[LongTensor, None, None]:
        """Causal sampling from LLM by utilising KV Cache

        Args:
            input_ids (LongTensor): B x S input tokens
            stop_token (int): stop token id
            max_tokens (int, optional): max tokens to generate. Defaults to 200.

        Yields:
            Generator[LongTensor, None, None]: generated tokens
        """

        generated_token_ids = None
        # generated_token_ids: B x 1

        token_countdown = max_tokens

        while token_countdown > 0:
            x: Tensor = self.embed_tokens(input_ids)
            # x: B x S x d

            x = self.decoder(x, use_cache=True)
            # x: B x S x d

            logits = self.lm_head(x[:, -1, :])
            # logits: B x C

            probs = F.softmax(logits, dim=1)
            # probs: B x C

            generated_token_ids = torch.multinomial(probs, num_samples=1)
            # generated_token_ids: B x 1

            yield generated_token_ids

            if (generated_token_ids == stop_token).all():
                # If all generated stop tokens, end the loop.
                break

            input_ids = generated_token_ids
            # input_ids: B x 1

            token_countdown -= 1

    def reset_cache(self):
        # Resets the KV Caches
        [layer.self_attn.reset_cache() for layer in self.layers]

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
    model.eval()
    total_loss = []
    for batch, targets, padding_mask in tqdm(dl):
        logits = model(batch.to(device), padding_mask=padding_mask.to(device))
        loss = criterion(
            logits.flatten(start_dim=0, end_dim=1),
            targets.flatten(start_dim=0, end_dim=1).to(device),
        )
        total_loss.append(loss.item())
    return sum(total_loss) / len(total_loss)


def inference_loop(
    model: SmolLM2,
    dataset: InstructDataset,
    device: str,
    num_infer_samples: int = 3,
    # TODO add if user messages needs to be inferred or inserted
):
    model.eval()

    tokenizer = dataset._tokenizer

    # get random samples
    data_ids = list(range(len(dataset)))
    for idx in random.sample(data_ids, k=num_infer_samples):
        messages: List[Message] = deepcopy(dataset._data[idx])

        system_message = messages[0]
        user_messages = filter(lambda message: message.role == "user", messages)
        current_messages: List[Message] = [system_message]
        message_offset: int = 0
        max_token = max(
            [
                len(tokenizer.encode(message.content))
                for message in messages
                if message.role == "assistant"
            ]
        )
        user_messages = list(user_messages)
        user_messages = user_messages + user_messages

        for user_message in user_messages:
            current_messages.append(user_message)
            input_raw: str = tokenizer.apply_chat_template(current_messages)

            input_delta_raw: str = input_raw[message_offset:]
            message_offset += len(input_delta_raw)

            print(input_delta_raw, flush=True)

            input_ids: LongTensor = torch.tensor(
                [tokenizer.encode(input_delta_raw)],
                dtype=torch.long,
                device=device,
            )
            # input_ids: 1, S

            assistant_message = Message(role="assistant", content="")
            for token_id in model.sample(
                input_ids,
                tokenizer.eos_token_id,
                max_tokens=max_token,
            ):
                token = tokenizer.decode([token_id.item()])
                assistant_message.content += token
                print(token, flush=True, end="")

            message_offset += len(assistant_message.content)
            current_messages.append(assistant_message)
        model.reset_cache()
        print("\n", "---" * 20, "\n")


def main(args):
    model_name: str = args.model_name
    data_path: str = args.data_path
    artifacts_path: str = args.artifacts_path
    tokenizer = Tokenizer.from_json_file(
        os.path.join(artifacts_path, f"{model_name}-tokenizer.json")
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

    model_ckpt_file_path: str = os.path.join(
        artifacts_path, f"{model_name}-best-full-sft.ckpt"
    )
    model_original_ckpt_file_path: str = os.path.join(
        artifacts_path, f"{model_name}-it.ckpt"
    )

    if args.resume:
        print(f"Resuming from {model_ckpt_file_path}")
        model = SmolLM2.from_checkpoint(model_ckpt_file_path)
    else:
        model = SmolLM2.from_checkpoint(model_original_ckpt_file_path)

    if is_multi_gpu_training:
        print(f"{num_available_gpus} GPUs found, switching to DDP")
        multi_gpu_config = MultiGPUConfig()
        torch.cuda.set_device(multi_gpu_config.local_rank)
        ddp_setup(multi_gpu_config.rank, multi_gpu_config.world_size)
        device: str = f"cuda:{multi_gpu_config.local_rank}"
        model.to(device)
        model = DDP(model, device_ids=[multi_gpu_config.local_rank])

    is_master = (not is_multi_gpu_training) or (torch.distributed.get_rank() == 0)

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
        model.parameters(),
        lr=learning_rate,
    )

    best_val_loss = math.inf
    num_infer_samples: int = args.num_infer_samples

    for i in range(epoch):
        if is_master:
            print(
                f"running random {num_infer_samples} samples from validation set to sanity check"
            )
            inference_loop(model, val_ds, device, num_infer_samples=num_infer_samples)

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

            if is_multi_gpu_training:
                config = asdict(model.module.config)
                weights = model.module.state_dict()
            else:
                config = asdict(model.config)
                weights = model.state_dict()

            torch.save(
                {"config": config, "weights": weights},
                model_ckpt_file_path,
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
        choices=["smollm2-135m", "smollm2-360m"],
        required=True,
    )
    ap.add_argument("--data-path", "-dp", type=str, default="data")
    ap.add_argument("--artifacts-path", "-ap", type=str, default="artifacts")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", "-d", type=str, default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--num-process", "-ns", type=int, default=16)
    ap.add_argument("--learning-rate", "-lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", "-bs", type=int, default=32)
    ap.add_argument("--epoch", "-e", type=int, default=10)
    ap.add_argument("--ignore-index", "-ig", type=int, default=-100)
    ap.add_argument("--num-infer-samples", "-ni", type=int, default=3)

    main(ap.parse_args())
