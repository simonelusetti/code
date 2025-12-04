from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, load_dataset, load_from_disk
from dora import to_absolute_path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

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
    }
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
    "ADJ": 1,  # ACTION
}


def map_pos_to_group(pos: str) -> int:
    return POS_GROUP_MAP.get(pos, 2)  # OTHER


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
    dataset_config: dict = DATASETS_DEFAULT_CONFIG["wikiann"]
) -> str:
    canonical = canonical_name(name)
    parts = [canonical, split]
    config_values = dataset_config.values()
    for value in config_values:
        if value is not None:
            parts.append(sanitize_fragment(str(value)))
    filename = "_".join(parts)
    return filename


def dataset_path(
    name: str,
    split: str,
    dataset_config: dict = DATASETS_DEFAULT_CONFIG["wikiann"]
) -> str:
    filename = dataset_filename(name, split, dataset_config)
    return to_absolute_path(f"./data/cache/{filename}") 


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
    ds = ds.select(range(target_subset))
    return ds


def resolve_dataset(
    name: str,
    split: str,
    dataset_config: dict = DATASETS_DEFAULT_CONFIG["wikiann"],
    raw_dataset_path: Optional[str],
) -> Tuple[Dataset, Callable]:
    name = canonical_name(name)
    raw_split_path = None
    if raw_dataset_path is not None:
        raw_split_path = Path(raw_dataset_path) / split
        if not raw_split_path.exists():
            raise FileNotFoundError(f"Raw dataset split not found at {raw_split_path}")

    if name == "cnn_dailymail":
        field = dataset_config["field"] or DATASETS_DEFAULT_CONFIG["cnn_dailymail"]["field"]
        ds = load_from_disk(str(raw_split_path)) if raw_split_path is not None else load_dataset("cnn_dailymail", field, split=split)
        target_field = field or "highlights"
        return ds, lambda x: x[target_field]
    elif name == "wikiann":
        config_name = dataset_config["language"] or DATASETS_DEFAULT_CONFIG["wikiann"]["language    "]
        ds = load_from_disk(str(raw_split_path)) if raw_split_path is not None else load_dataset("wikiann", config_name, split=split)
        return ds, lambda x: x["tokens"]
    elif name == "conll2003":
        ds = load_from_disk(str(raw_split_path)) if raw_split_path is not None else load_dataset("conll2003", split=split)
        return ds, lambda x: x["tokens"]
    elif name == "ud":
        text_fn = lambda x: x["form"]
        if raw_split_path is not None:
            ds = load_from_disk(raw_split_path)
            return ds, text_fn

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

        def load_conllu(path: Path | str) -> list[dict[str, list]]:
            sentences = []
            with open(path, "r", encoding="utf-8") as f:
                for sent in parse_incr(f):
                    tokens = [tok["form"] for tok in sent]
                    upos = [tok["upos"] for tok in sent]
                    labels = [map_pos_to_group(p) for p in upos]
                    sentences.append({"tokens": tokens, "labels": labels})
            return sentences
            
        datasets_dict = {split: load_conllu(path) for split, path in split_paths.items()}
        ds = DatasetDict({split: Dataset.from_list(data) for split, data in datasets_dict.items()})
        return ds[split], text_fn
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    return ds, text_fn


def encode_examples(
    name: str,
    ds: Dataset,
    tok: AutoTokenizer,
    encoder: AutoModel,
    text_fn: Callable,
    max_length: int,
) -> Dataset:
    def _tokenize_and_encode(example):
        text = text_fn(example)
        split_into_words = isinstance(text, (list, tuple))
        enc = tok(
            text,
            truncation=True,
            max_length=max_length,
            is_split_into_words=split_into_words,
        )

        device = next(encoder.parameters()).device
        inputs = {
            "input_ids": torch.tensor(enc["input_ids"], device=device).unsqueeze(0),
            "attention_mask": torch.tensor(enc["attention_mask"], device=device).unsqueeze(0),
        }
        with torch.no_grad():
            out = encoder(**inputs, output_attentions=False, return_dict=True)

        out_dict = {
            "input_ids": np.asarray(enc["input_ids"], dtype=np.int64),
            "attention_mask": np.asarray(enc["attention_mask"], dtype=np.int64),
            "embeddings": out.last_hidden_state.squeeze(0).detach().cpu().to(torch.float32).numpy(),
            "tokens": example.get("tokens", example["form"])
        }

        if "labels" in example and split_into_words:
            word_ids = enc.word_ids()
            labels = tok["upos"]
            aligned = []
            for word_id in word_ids:
                if word_id is None:
                    aligned.append(0)
                else:
                    aligned.append(map_pos_to_group(p) for p in upos[word_id])
            out_dict["factor_tags"] = np.asarray(aligned, dtype=np.int64)

        if "ner_tags" in example and split_into_words:
            word_ids = enc.word_ids()
            ner_tags = example["ner_tags"]
            aligned = []
            for word_id in word_ids:
                if word_id is None:
                    aligned.append(0)
                else:
                    aligned.append(0 if NER_FALSE_NAMES["name"] == ner_tags[word_id] else 1)
            out_dict["ner_tags"] = np.asarray(aligned, dtype=np.int64)
            
        return out_dict

    return ds.map(_tokenize_and_encode, remove_columns=ds.column_names, batched=False)


def build_dataset(
    name: str,
    tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    dataset_config: dict = DATASETS_DEFAULT_CONFIG["wikiann"],
    split: str = "train",
    subset: int | float | None = None,
    max_length: int = 512,
    shuffle: bool = False,
    raw_dataset_path: Optional[str] = None,
) -> Tuple[Dataset, AutoTokenizer]:
    name = canonical_name(name)
    full_ds, text_fn = resolve_dataset(
        name=name,
        split=split,
        dataset_config=dataset_config,
        raw_dataset_root=raw_dataset_root,
        cnn_field=cnn_field,
    )
    ds = apply_subset_and_shuffle(full_ds, subset, shuffle=shuffle, log=log)
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if name == "ud":
        return ds, tok
    encoder = freeze_encoder(AutoModel.from_pretrained(tokenizer_name))
    ds = encode_examples(name, ds, tok, encoder, text_fn, max_length)
    return ds, tok


def get_dataset(
    tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    name: str = "wikiann",
    split: str = "train",
    dataset_config: dict = DATASETS_DEFAULT_CONFIG["wikiann"],
    subset: int | float | None = None,
    rebuild: bool = False,
    shuffle: bool = False,
    log: bool = True,
    max_length: int = 512,
    raw_dataset_path: Optional[str] = None,
) -> Dataset:
    name = canonical_name(name)
    cache_path = dataset_cache_paths(
        name,
        split,
        subset,
        cnn_field=cnn_field,
        dataset_config=dataset_config,
    )
    if cache_path.exists() and not rebuild:
        if log:
            logger.info("Loading cached dataset from %s", cache_path)
        ds = load_from_disk(cache_path)
        tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if shuffle and subset not in (None, 1.0):
            ds = subset_and_shuffle(ds, subset, shuffle=shuffle)
        return ds
    if log:
        logger.info("Building dataset for %s/%s (subset=%s)", name, split, subset)
    ds, tok = build_dataset(
        name=name,
        split=split,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        subset=subset,
        shuffle=shuffle,
        dataset_config=dataset_config,
        raw_dataset_path=raw_dataset_path,
        log=log,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(cache_path)
    return ds


def collate(batch):
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
    embeddings = pad_sequence(
        tuple(_as_tensor(item["embeddings"], torch.float) for item in batch),
        batch_first=True,
        padding_value=0.0,
    )
    ner_tags = None
    if "ner_tags" in batch[0]:
        ner_tags = pad_sequence(
            tuple(_as_tensor(item["ner_tags"], torch.long) for item in batch),
            batch_first=True,
            padding_value=-100,
        )
    factor_tags = None
    if "labels" in batch[0]:
        factor_tags = pad_sequence(
            tuple(_as_tensor(item["labels"], torch.long) for item in batch),
            batch_first=True,
            padding_value=2,  # OTHER
        )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "factor_tags": factor_tags,
        "ner_tags": ner_tags,
        "embeddings": embeddings,
    }


def initialize_dataloaders(cfg, log: bool = True):
    train_cfg = cfg.data.train
    eval_cfg = cfg.data.eval
    dev_cfg = cfg.data.dev

    tokenizer_name = cfg.bucket_model.sbert_model

    def build_loader(split_cfg):
        split = split_cfg.split
        ds = get_dataset(
            tokenizer_name=tokenizer_name,
            name=_canonical_name(split_cfg.dataset),
            split=split,
            subset=split_cfg.subset,
            dataset_config=split_cfg.config,
            rebuild=cfg.data.rebuild_ds,
            shuffle=split_cfg.shuffle,
            log=log,
            raw_dataset_path=None,
        )
        return DataLoader(
            ds,
            batch_size=split_cfg.batch_size,
            collate_fn=collate,
            num_workers=split_cfg.num_workers,
            pin_memory=(cfg.runtime.device == "cuda"),
            persistent_workers=(split_cfg.num_workers > 0),
            shuffle=split_cfg.shuffle,
        )

    train_dl = build_loader(train_cfg)
    eval_dl = build_loader(eval_cfg)

    dev_dl = None
    if dev_cfg and dev_cfg.dataset:
        dev_dl = build_loader(dev_cfg)

    return train_dl, eval_dl, dev_dl
