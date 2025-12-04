from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence
from urllib.request import urlretrieve

import numpy as np
import torch
from conllu import parse_incr
from datasets import Dataset, DatasetDict, load_from_disk
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer

from .utils import freeze_encoder

logger = logging.getLogger(__name__)

# POS → 3-class mapping
POS_GROUP_MAP = {
    "NOUN": 0,
    "PROPN": 0,
    "PRON": 0,  # THING
    "VERB": 1,
    "AUX": 1,
    "ADV": 1,
    "ADJ": 1,  # ACTION
}


def map_pos_to_group(pos: str) -> int:
    return POS_GROUP_MAP.get(pos, 2)  # OTHER


def load_conllu(path: Path | str) -> list[dict[str, list]]:
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for sent in parse_incr(f):
            tokens = [tok["form"] for tok in sent]
            upos = [tok["upos"] for tok in sent]
            labels = [map_pos_to_group(p) for p in upos]
            sentences.append({"tokens": tokens, "labels": labels})
    return sentences


def _pad_to_length(seqs: Sequence[Sequence[int]], pad_id: int) -> list[list[int]]:
    max_len = max(len(seq) for seq in seqs)
    return [list(seq) + [pad_id] * (max_len - len(seq)) for seq in seqs]


def _encode_batch(
    batch: Mapping[str, Sequence[Sequence]],
    tokenizer: AutoTokenizer,
    encoder: AutoModel,
    max_length: int = 256,
) -> dict[str, list]:
    enc = tokenizer(
        batch["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
        return_attention_mask=True,
    )

    aligned_labels: list[list[int]] = []
    for i, labels in enumerate(batch["labels"]):
        word_ids = enc.word_ids(batch_index=i)
        expanded = []
        for w in word_ids:
            if w is None:
                expanded.append(2)  # OTHER for special tokens
            else:
                expanded.append(int(labels[w]))
        aligned_labels.append(expanded)

    padded_ids = _pad_to_length(enc["input_ids"], tokenizer.pad_token_id or 0)
    padded_masks = _pad_to_length(enc["attention_mask"], 0)
    padded_labels = _pad_to_length(aligned_labels, 2)  # OTHER for padding

    device = next(encoder.parameters()).device
    input_ids_tensor = torch.tensor(padded_ids, device=device)
    attention_tensor = torch.tensor(padded_masks, device=device)

    with torch.no_grad():
        outputs = encoder(
            input_ids_tensor,
            attention_mask=attention_tensor,
            return_dict=True,
        )
    embeddings = outputs.last_hidden_state.detach().cpu().numpy()

    return {
        "input_ids": padded_ids,
        "attention_mask": padded_masks,
        "embeddings": embeddings,
        "labels": padded_labels,
    }


def build_ud_sbert_dataset(
    tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_length: int = 256,
    cache_dir: str | Path = "./data/ud-ewt",
    download_missing: bool = True,
) -> tuple[DatasetDict, AutoTokenizer]:
    cache_dir = Path(cache_dir)
    if cache_dir.exists():
        ds = DatasetDict.load_from_disk(cache_dir) if hasattr(DatasetDict, "load_from_disk") else load_from_disk(cache_dir)
        tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        return ds, tok

    files = {
        "train": "en_ewt-ud-train.conllu",
        "validation": "en_ewt-ud-dev.conllu",
        "test": "en_ewt-ud-test.conllu",
    }
    base_url = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/"

    cache_dir.mkdir(parents=True, exist_ok=True)
    split_paths: dict[str, Path] = {}
    for split, fname in files.items():
        local = cache_dir / fname
        split_paths[split] = local
        if not local.exists():
            if not download_missing:
                raise FileNotFoundError(f"Missing UD file {local}")
            logger.info("Downloading UD %s split to %s", split, local)
            urlretrieve(base_url + fname, local)

    datasets_dict = {split: load_conllu(path) for split, path in split_paths.items()}
    hf_ds = DatasetDict({split: Dataset.from_list(data) for split, data in datasets_dict.items()})

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    encoder = freeze_encoder(AutoModel.from_pretrained(tokenizer_name))

    encoded = hf_ds.map(
        lambda b: _encode_batch(b, tokenizer, encoder, max_length=max_length),
        batched=True,
        batch_size=4,
        remove_columns=["tokens", "labels"],
    )

    encoded.save_to_disk(cache_dir)
    return encoded, tokenizer


def collate_ud(batch):
    def _as_tensor(value, dtype):
        if isinstance(value, torch.Tensor):
            return value.to(dtype=dtype)
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).to(dtype=dtype)
        return torch.tensor(value, dtype=dtype)

    input_ids = pad_sequence(
        tuple(_as_tensor(item["input_ids"], torch.long) for item in batch),
        batch_first=True,
        padding_value=0,
    )
    attention_masks = pad_sequence(
        tuple(_as_tensor(item["attention_mask"], torch.long) for item in batch),
        batch_first=True,
        padding_value=0,
    )
    labels = pad_sequence(
        tuple(_as_tensor(item["labels"], torch.long) for item in batch),
        batch_first=True,
        padding_value=2,  # OTHER
    )
    embeddings = pad_sequence(
        tuple(_as_tensor(item["embeddings"], torch.float) for item in batch),
        batch_first=True,
        padding_value=0.0,
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
        "embeddings": embeddings,
    }
