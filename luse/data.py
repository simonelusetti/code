from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from dora import to_absolute_path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from urllib.request import urlretrieve
from conllu import parse_incr

# ---------------------------------------------------------------------------
# Constants & mappings
# ---------------------------------------------------------------------------

ALIASES: dict[str, set[str]] = {
    "cnn_dailymail": {"cnn", "cnn_highlights", "cnn_dailymail"},
    "wikiann": {"wikiann", "wikiann_en"},
    "conll2003": {"conll2003", "conll"},
    "ud": {"ud"},
}

DATASETS_DEFAULT_CONFIG = {
    "wikiann": {
        "language": "en",
    },
    "cnn_dailymail": {
        "version": "3.0.0",
        "field": "highlights",
    },
    "conll2003": None,
}

NER_FALSE_NAMES = {
    "wikiann": "O",
    "conll2003": "O",
}

# POS → 3-class mapping
POS_GROUP_MAP = {
    "NOUN": 0,
    "PROPN": 0,
    "PRON": 0,  # THING
    "VERB": 1,
    "AUX": 1,
    "ADV": 1,
    "ADJ": 1,   # ACTION
}


def map_pos_to_group(pos: str) -> int:
    return POS_GROUP_MAP.get(pos, 2)  # OTHER


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def sanitize_fragment(fragment: str) -> str:
    return fragment.replace("/", "-")


def canonical_name(name: str) -> str:
    for canonical, aliases in ALIASES.items():
        if name == canonical or name in aliases:
            return canonical
    return name


def dataset_filename(
    name: str,
    split: str,
    dataset_config: dict = DATASETS_DEFAULT_CONFIG["wikiann"],
) -> str:
    canonical = canonical_name(name)
    parts = [canonical, split]
    if dataset_config is None:
        return "_".join(parts)
    for value in dataset_config.values():
        if value is not None:
            parts.append(sanitize_fragment(str(value)))
    return "_".join(parts)


def dataset_path(
    name: str,
    split: str,
    dataset_config: dict = DATASETS_DEFAULT_CONFIG["wikiann"],
) -> Path:
    filename = dataset_filename(name, split, dataset_config)
    return Path(to_absolute_path(f"./data/cache/{filename}"))


def freeze_encoder(encoder: AutoModel) -> AutoModel:
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder


def subset_and_shuffle(ds: Dataset, subset, shuffle: bool) -> Dataset:
    if shuffle:
        ds = ds.shuffle(seed=42)
    if subset is None or subset == 1.0:
        return ds
    target_subset = int(len(ds) * subset) if subset <= 1.0 else int(subset)
    if target_subset <= 0:
        raise ValueError(f"Requested subset {subset} results in 0 examples.")
    return ds.select(range(target_subset))


# ---------------------------------------------------------------------------
# Dataset resolution (raw → HF Dataset)
# ---------------------------------------------------------------------------

def resolve_dataset(
    name: str,
    split: str,
    dataset_config: dict = DATASETS_DEFAULT_CONFIG["wikiann"],
    raw_dataset_path: Optional[str] = None,
) -> Tuple[Dataset, Callable]:
    """
    Returns (dataset, text_fn) where text_fn(example) -> tokens or text.
    """
    name = canonical_name(name)

    raw_split_path = None
    if raw_dataset_path is not None:
        raw_root = Path(raw_dataset_path)
        raw_split_path = raw_root / split
        if not raw_split_path.exists() and name != "ud":
            raise FileNotFoundError(f"Raw dataset split not found at {raw_split_path}")

    if name == "cnn_dailymail":
        version = dataset_config.get("version") or DATASETS_DEFAULT_CONFIG["cnn_dailymail"]["version"]
        ds = load_from_disk(str(raw_split_path)) if raw_split_path is not None else load_dataset("cnn_dailymail", version, split=split)
        target_field = dataset_config.get("field") or DATASETS_DEFAULT_CONFIG["cnn_dailymail"]["field"]
        return ds, lambda x: x[target_field]

    if name == "wikiann":
        config_name = dataset_config.get("language") or DATASETS_DEFAULT_CONFIG["wikiann"]["language"]
        ds = load_from_disk(str(raw_split_path)) if raw_split_path is not None else load_dataset("wikiann", config_name, split=split)
        return ds, lambda x: x["tokens"]

    if name == "conll2003":
        ds = load_from_disk(str(raw_split_path)) if raw_split_path is not None else load_dataset("conll2003", split=split)
        return ds, lambda x: x["tokens"]

    if name == "ud":
        text_fn = lambda x: x["tokens"]

        if raw_split_path is not None:
            ds = load_from_disk(raw_split_path)
            return ds, text_fn

        files = {
            "train": "en_ewt-ud-train.conllu",
            "validation": "en_ewt-ud-dev.conllu",
            "test": "en_ewt-ud-test.conllu",
        }
        base_url = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/"

        raw_root = Path(to_absolute_path("./data/raw/ud"))
        raw_root.mkdir(parents=True, exist_ok=True)

        split_paths: dict[str, Path] = {}
        for sp, fname in files.items():
            local = raw_root / fname
            split_paths[sp] = local
            if not local.exists():
                urlretrieve(base_url + fname, local)

        def load_conllu(path: Path | str) -> list[dict[str, list]]:
            sentences = []
            with open(path, "r", encoding="utf-8") as f:
                for sent in parse_incr(f):
                    tokens = [tok["form"] for tok in sent]
                    upos = [tok["upos"] for tok in sent]
                    labels = [map_pos_to_group(p) for p in upos]
                    sentences.append({"tokens": tokens, "labels": labels})
            return sentences

        datasets_dict = {sp: load_conllu(path) for sp, path in split_paths.items()}
        ds = DatasetDict({sp: Dataset.from_list(data) for sp, data in datasets_dict.items()})
        return ds[split], text_fn

    raise ValueError(f"Unknown dataset name: {name}")


# ---------------------------------------------------------------------------
# Encoding with frozen encoder (precompute embeddings)
# ---------------------------------------------------------------------------

def encode_examples(
    name: str,
    ds: Dataset,
    tok: AutoTokenizer,
    encoder: AutoModel,
    text_fn: Callable,
    max_length: int,
) -> Dataset:
    """
    Maps each example to:
      - input_ids: List[int]
      - attention_mask: List[int]
      - embeddings: List[List[float]] (last_hidden_state)
      - tokens: List[str] (BERT wordpiece tokens)
      - factor_tags / ner_tags: List[int], aligned to wordpieces
    All stored as Python lists (no numpy), tensorized later in collate().
    """

    device = next(encoder.parameters()).device

    def _tokenize_and_encode(example):
        text = text_fn(example)
        split_into_words = isinstance(text, (list, tuple))

        enc = tok(
            text,
            truncation=True,
            max_length=max_length,
            is_split_into_words=split_into_words,
        )

        # to tensors on encoder device
        input_ids = torch.tensor(enc["input_ids"], device=device).unsqueeze(0)
        attention_mask = torch.tensor(enc["attention_mask"], device=device).unsqueeze(0)

        with torch.no_grad():
            out = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=False,
                return_dict=True,
            )

        # move to CPU, keep as Python lists (no numpy)
        last_hidden = out.last_hidden_state.squeeze(0).cpu().to(torch.float32)

        out_dict = {
            "input_ids": enc["input_ids"],                           # List[int]
            "attention_mask": enc["attention_mask"],                 # List[int]
            "embeddings": last_hidden.tolist(),                      # List[List[float]]
            "tokens": tok.convert_ids_to_tokens(enc["input_ids"]),   # BERT wordpieces
        }

        # POS / factor tags
        if "labels" in example and split_into_words:
            word_ids = enc.word_ids()
            labels = example["labels"]  # word-level labels 0/1/2
            aligned = []
            for wid in word_ids:
                if wid is None:
                    aligned.append(2)  # OTHER
                else:
                    aligned.append(int(labels[wid]))
            out_dict["factor_tags"] = aligned  # List[int]

        # NER tags: binary entity vs non-entity
        if "ner_tags" in example and split_into_words:
            word_ids = enc.word_ids()
            ner_tags = example["ner_tags"]
            aligned = []
            false_name = NER_FALSE_NAMES.get(name, "O")
            for wid in word_ids:
                if wid is None:
                    aligned.append(0)
                else:
                    aligned.append(0 if false_name == ner_tags[wid] else 1)
            out_dict["ner_tags"] = aligned  # List[int]

        return out_dict

    return ds.map(_tokenize_and_encode, remove_columns=ds.column_names, batched=False)


# ---------------------------------------------------------------------------
# Public dataset builders
# ---------------------------------------------------------------------------

def build_dataset(
    name: str,
    tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    dataset_config: dict = DATASETS_DEFAULT_CONFIG["wikiann"],
    split: str = "train",
    subset: int | float | None = None,
    max_length: int = 512,
    shuffle: bool = False,
    raw_dataset_path: Optional[str] = None,
) -> Dataset:
    name = canonical_name(name)
    full_ds, text_fn = resolve_dataset(
        name=name,
        split=split,
        dataset_config=dataset_config,
        raw_dataset_path=raw_dataset_path,
    )
    ds = subset_and_shuffle(full_ds, subset, shuffle=shuffle)

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    encoder = freeze_encoder(AutoModel.from_pretrained(tokenizer_name))

    ds = encode_examples(name, ds, tok, encoder, text_fn, max_length)
    return ds


def get_dataset(
    tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    name: str = "wikiann",
    split: str = "train",
    dataset_config: dict = DATASETS_DEFAULT_CONFIG["wikiann"],
    subset: int | float | None = None,
    rebuild: bool = False,
    shuffle: bool = False,
    logger: Optional[logging.Logger] = None,
    max_length: int = 512,
    raw_dataset_path: Optional[str] = None,
) -> Dataset:
    name = canonical_name(name)
    path = dataset_path(name, split, dataset_config=dataset_config)

    if path.exists() and not rebuild:
        if logger is not None:
            logger.info("Loading cached dataset from %s", path)
        ds = load_from_disk(path)
        if shuffle and subset not in (None, 1.0):
            ds = subset_and_shuffle(ds, subset, shuffle=shuffle)
        return ds

    if logger is not None:
        logger.info("Building dataset for %s/%s (subset=%s)", name, split, subset)

    ds = build_dataset(
        name=name,
        tokenizer_name=tokenizer_name,
        dataset_config=dataset_config,
        split=split,
        subset=subset,
        max_length=max_length,
        shuffle=shuffle,
        raw_dataset_path=raw_dataset_path,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(path)
    return ds


# ---------------------------------------------------------------------------
# Collate function (everything tensorized here)
# ---------------------------------------------------------------------------

def collate(batch):
    """
    batch: list[dict]
    Returns tensors:
      - input_ids:      (B, L)
      - attention_mask: (B, L)
      - embeddings:     (B, L, D)
      - ner_tags:       (B, L) or None
      - factor_tags:    (B, L) or None
      - tokens:         list[list[str]] (kept as Python, for SBERT / gating)
    """

    def _to_tensor(value, dtype):
        if isinstance(value, torch.Tensor):
            return value.to(dtype=dtype)
        # value is list (nested or flat)
        return torch.tensor(value, dtype=dtype)

    input_ids = pad_sequence(
        [ _to_tensor(item["input_ids"], torch.long) for item in batch ],
        batch_first=True,
        padding_value=0,
    )

    attention_masks = pad_sequence(
        [ _to_tensor(item["attention_mask"], torch.long) for item in batch ],
        batch_first=True,
        padding_value=0,
    )

    embeddings = pad_sequence(
        [ _to_tensor(item["embeddings"], torch.float32) for item in batch ],
        batch_first=True,
        padding_value=0.0,
    )

    ner_tags = None
    if "ner_tags" in batch[0]:
        ner_tags = pad_sequence(
            [ _to_tensor(item["ner_tags"], torch.long) for item in batch ],
            batch_first=True,
            padding_value=-100,  # ignore_index
        )

    factor_tags = None
    if "factor_tags" in batch[0]:
        factor_tags = pad_sequence(
            [ _to_tensor(item["factor_tags"], torch.long) for item in batch ],
            batch_first=True,
            padding_value=-100,  # OTHER / ignore_index
        )

    tokens = [item["tokens"] for item in batch]  # list[list[str]]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "factor_tags": factor_tags,
        "ner_tags": ner_tags,
        "embeddings": embeddings,
        "tokens": tokens,
    }


# ---------------------------------------------------------------------------
# Dataloader initialization
# ---------------------------------------------------------------------------

def initialize_dataloaders(
    cfg: dict,
    logger: Optional[logging.Logger] = None,
    tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:

    def build_loader(split_cfg):
        ds = get_dataset(
            tokenizer_name=tokenizer_name,
            name=canonical_name(split_cfg.dataset),
            split=split_cfg.split,
            subset=split_cfg.subset,
            dataset_config=split_cfg.config,
            rebuild=cfg.rebuild_ds,
            shuffle=split_cfg.shuffle,
            logger=logger,
            max_length=cfg.max_length,
            raw_dataset_path=None,
        )
        return DataLoader(
            ds,
            batch_size=split_cfg.batch_size,
            collate_fn=collate,
            num_workers=split_cfg.num_workers,
            pin_memory=(device == "cuda"),
            persistent_workers=(split_cfg.num_workers > 0),
            shuffle=split_cfg.shuffle,
        )

    train_dl = build_loader(cfg.train)
    eval_dl = build_loader(cfg.eval)

    dev_dl = None
    if getattr(cfg, "dev", None) is not None and cfg.dev.dataset:
        dev_dl = build_loader(cfg.dev)

    return train_dl, eval_dl, dev_dl
