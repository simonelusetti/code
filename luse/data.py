from __future__ import annotations

import logging, gzip, nltk
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from dora import to_absolute_path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from urllib.request import urlretrieve
from conllu import parse_incr
from nltk.corpus import brown, treebank


# ---------------------------------------------------------------------------
# Constants & mappings
# ---------------------------------------------------------------------------

ALIASES: dict[str, set[str]] = {
    "cnn_dailymail": {"cnn", "cnn_highlights", "cnn_dailymail"},
    "wikiann": {"wikiann", "wikiann_en"},
    "conll2003": {"conll2003", "conll03"},
    "conll2000": {"conll2000", "chunking", "conll00"},
    "ud": {"ud"},
    "brown": {"brown"},
    "treebank": {"treebank", "tb"},
}

ALIASES["parasci"] = {"parasci"}
ALIASES["parasci-concat"] = {"parasci-concat", "parasci_concat"}

DATASETS_DEFAULT_CONFIG = {
    "wikiann": {
        "language": "en",
    },
    "cnn_dailymail": {
        "version": "3.0.0",
        "field": "highlights",
    },
    "ud": None,
    "conll2003": None,
    "conll2000": None,
    "brown": None,
    "treebank": None,
}

DATASETS_DEFAULT_CONFIG["parasci"] = None
DATASETS_DEFAULT_CONFIG["parasci-concat"] = None

ALIASES["nps_chat"] = {"nps_chat","nps","chat","npschat"}
DATASETS_DEFAULT_CONFIG["nps_chat"] = None

NER_FALSE_NAMES = {
    "wikiann": "O",
    "conll2003": "O",
}

USES_SYSTEM = {"ud", "conll2000", "brown", "treebank", "nps_chat"}
POS_SYSTEMS = {
    "ud": {
        "NOUN": "thing", "PROPN": "thing", "PRON": "thing",
        "VERB": "action", "AUX": "action",
        "ADJ": "action",          # was "thing"
        "ADV": "action",
        "ADP": "syntax",
        "DET": "syntax",          # was "thing"
        "CCONJ": "syntax",
        "SCONJ": "syntax",
        "NUM": "thing",
        "PART": "syntax",
        "PUNCT": "syntax",
        "SYM": "syntax",
        "_": "syntax",
        "INTJ": "other",
        "X": "other",
    },
}

POS_SYSTEMS["spacy"] = POS_SYSTEMS["ud"]

POS_SYSTEMS["conll2000"] = {
    # THINGS
    "NN": "thing", "NNS": "thing", "NNP": "thing", "NNPS": "thing",
    "PRP": "thing", "PRP$": "thing",
    "WP": "thing", "WP$": "thing",

    # ACTIONS
    "VB": "action", "VBD": "action", "VBG": "action",
    "VBN": "action", "VBP": "action", "VBZ": "action",
    "JJ": "action", "JJR": "action", "JJS": "action",
    "RB": "action", "RBR": "action", "RBS": "action",
    "WRB": "action",

    # SYNTAX (true function words)
    "IN": "syntax",
    "DT": "syntax",
    "PDT": "syntax",
    "CC": "syntax",
    "MD": "syntax",
    "POS": "syntax",
    "RP": "syntax",
    "TO": "syntax",
    "WDT": "syntax",
    "``": "syntax", "''": "syntax",
    ",": "syntax", ".": "syntax", ":": "syntax",
    "(": "syntax", ")": "syntax",
    "#": "syntax", "$": "syntax",

    # OTHER (more semantically precise)
    "CD": "thing",    # was other
    "FW": "thing",    # was other
    "EX": "syntax",   # was other
    "LS": "syntax",   # was other
    "SYM": "other",
    "UH": "other",
}

POS_SYSTEMS["treebank"] = {
    **POS_SYSTEMS["conll2000"],
    "-NONE-": "other",
    "-LRB-": "syntax",
    "-RRB-": "syntax",
    "-LCB-": "syntax",
    "-RCB-": "syntax",
}

POS_SYSTEMS["brown"] = {
    "NOUN": "thing",
    "PROPN": "thing",
    "PRON": "thing",

    "VERB": "action",
    "AUX": "action",

    "ADJ": "action",
    "ADV": "action",

    "ADP": "syntax",
    "CONJ": "syntax",
    "SCONJ": "syntax",
    "PART": "syntax",

    "DET": "thing",
    "NUM": "thing",

    "PUNCT": "syntax",
    "PRT": "syntax",
    ".": "syntax",

    "SYM": "other",
    "INTJ": "other",
    "X": "other",
}

POS_SYSTEMS["nps_chat"] = {
    **POS_SYSTEMS["conll2000"],

    # Custom NPS Chat tags
    "EMO": "other",
    "URL": "other",
    "GW": "other",
    "HVS": "action",   # was other
    "X": "other",

    # Emphatic / uppercase → same as base
    "^NN":  "thing",
    "^NNS": "thing",
    "^NNP": "thing",

    "^JJ":  "action",
    "^JJR": "action",
    "^JJS": "action",

    "^RB":  "action",
    "^WRB": "action",

    "^VB":  "action",
    "^VBD": "action",
    "^VBN": "action",
    "^VBG": "action",
    "^VBP": "action",
    "^VBZ": "action",

    # Function words
    "^DT": "syntax",
    "^IN": "syntax",
    "^CC": "syntax",
    "^TO": "syntax",
    "^MD": "syntax",
    "^POS": "syntax",
    "^RP": "syntax",
    "^UH": "syntax",
    "^WP": "syntax",

    # Pronouns
    "^PRP":  "thing",
    "^PRP$": "thing",

    # Mixed
    "^PRP^VBP": "other",

    # caret punctuation
    "^.": "syntax",

    # Verb "BES"
    "BES": "action",

    # empty tag
    "": "other",
}

UNKNOWN_POS = set()

PART_TO_ID_BY_SYSTEM = {}
CATH_TO_ID = {"thing": 0, "action": 1, "other": 2, "syntax": 3, "pad": 4}

ID_TO_CATH = {idx: tag for tag, idx in CATH_TO_ID.items()}

for system_name, mapping in POS_SYSTEMS.items():
    tags = sorted(mapping.keys())
    local_map = {tag: i for i, tag in enumerate(tags)}
    local_map["pad"] = len(local_map)
    PART_TO_ID_BY_SYSTEM[system_name] = local_map

ID_TO_PART_BY_SYSTEM = {
    system: {idx: tag for tag, idx in tag_to_id.items()}
    for system, tag_to_id in PART_TO_ID_BY_SYSTEM.items()
}

def map_pos_to_group(pos: str, system: str) -> str:
    mapping = POS_SYSTEMS.get(system, {})
    if pos in mapping:
        return mapping[pos]
    UNKNOWN_POS.add((system, pos))

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


def subset_and_shuffle(ds: Dataset, subset: int | float = 1.0, shuffle: bool = False) -> Dataset:
    if shuffle:
        ds = ds.shuffle(seed=42)
    if subset is None or subset == 1.0:
        return ds
    target_subset = int(len(ds) * subset) if subset <= 1.0 else int(subset)
    if target_subset <= 0:
        raise ValueError(f"Requested subset {subset} results in 0 examples.")
    return ds.select(range(target_subset))


def load_nltk_pos_corpus(
    corpus_name: str,
    corpus_loader: Callable[[], list[list[tuple[str, str]]]],
    pos_system: str,
) -> Dataset:
    """
    Generic loader for NLTK POS-tagged corpora.
    Args:
        corpus_name: name to show in errors / download messages
        corpus_loader: function that returns a list of tagged sentences:
                       e.g. lambda: brown.tagged_sents(tagset="universal")
        pos_system: key into POS_SYSTEMS (e.g. "brown", "treebank", "nps_chat")
    Returns:
        A HuggingFace Dataset with {tokens, part_tags, cath_tags}
    """
    import nltk
    try:
        tagged = corpus_loader()
        if not tagged:
            raise LookupError
    except LookupError:
        nltk.download(corpus_name)
        tagged = corpus_loader()

    sentences = []
    for sent in tagged:
        tokens = [w for (w, _) in sent]
        pos_tags = [t for (_, t) in sent]
        cath_tags = [map_pos_to_group(t, pos_system) for t in pos_tags]

        sentences.append({
            "tokens": tokens,
            "part_tags": pos_tags,
            "cath_tags": cath_tags,
        })

    return Dataset.from_list(sentences)


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
        version = dataset_config.get("version") if dataset_config is not None else DATASETS_DEFAULT_CONFIG["cnn_dailymail"]["version"]
        ds = load_from_disk(str(raw_split_path)) if raw_split_path is not None else load_dataset("cnn_dailymail", version, split=split)
        target_field = dataset_config.get("field") if dataset_config is not None else DATASETS_DEFAULT_CONFIG["cnn_dailymail"]["field"]
        return ds, lambda x: x[target_field]

    if name == "wikiann":
        config_name = dataset_config.get("language") if dataset_config is not None else DATASETS_DEFAULT_CONFIG["wikiann"]["language"]
        ds = load_from_disk(str(raw_split_path)) if raw_split_path is not None else load_dataset("wikiann", config_name, split=split)
        return ds, lambda x: x["tokens"]

    if name == "conll2003":
        ds = load_from_disk(str(raw_split_path)) if raw_split_path is not None else load_dataset("conll2003", split=split)
        return ds, lambda x: x["tokens"]
    
    if name == "parasci":
        ds = load_from_disk(to_absolute_path("./data/cache/parasci"))
        return ds[split], lambda x: x["text"]

    if name == "parasci-concat":
        ds = load_from_disk(to_absolute_path("./data/cache/parasci-concat"))
        return ds[split], lambda x: x["text"]

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
                    part_labels = [tok["upos"] for tok in sent]
                    cath_labels = [map_pos_to_group(pos, "ud") for pos in part_labels]
                    sentences.append({
                        "tokens": tokens,
                        "cath_tags": cath_labels,
                        "part_tags": part_labels
                    })
            return sentences

        datasets_dict = {sp: load_conllu(path) for sp, path in split_paths.items()}
        ds = DatasetDict({sp: Dataset.from_list(data) for sp, data in datasets_dict.items()})
        return ds[split], text_fn
    
    if name == "conll2000":
        # Download  
        base_url = "https://www.clips.uantwerpen.be/conll2000/chunking/"
        files = {
            "train": "train.txt.gz",
            "validation": "test.txt.gz",
            "test": "test.txt.gz",
        }
        raw_root = Path(to_absolute_path("./data/raw/conll2000"))
        raw_root.mkdir(parents=True, exist_ok=True)
        gz_path = raw_root / files[split]
        txt_path = raw_root / files[split].replace(".gz", "")
        if not gz_path.exists():
            urlretrieve(base_url + files[split], gz_path)
        if not txt_path.exists():
            with gzip.open(gz_path, "rt", encoding="utf-8") as f_in, \
                 open(txt_path, "w", encoding="utf-8") as f_out:
                for line in f_in:
                    f_out.write(line)
        # Parse
        def _format_sentence(tokens, pos_tags):
            cath = [map_pos_to_group(p, "conll2000") for p in pos_tags]
            return{
                "tokens": tokens,
                "part_tags": pos_tags,
                "cath_tags": cath,
            }
            
        tokens, pos_tags, sentences = [], [], []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    if len(tokens) == 0:
                        raise ValueError("Empty sentence detected in CoNLL-2000 file.")
                    sentences.append(_format_sentence(tokens, pos_tags))
                    tokens, pos_tags = [], []
                    continue
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(f"Invalid CoNLL-2000 line (expected 3 columns): {line}")
                word, pos, _ = parts
                tokens.append(word)
                pos_tags.append(pos)
        if len(tokens) > 0:
            sentences.append(_format_sentence(tokens, pos_tags))

        ds = Dataset.from_list(sentences)
        return ds, lambda x: x["tokens"]
    
    if name == "brown":
        from nltk.corpus import brown
        ds = load_nltk_pos_corpus(
            corpus_name="brown",
            corpus_loader=lambda: brown.tagged_sents(tagset="universal"),
            pos_system="brown",
        )
        return ds, lambda x: x["tokens"]

    if name == "treebank":
        from nltk.corpus import treebank
        ds = load_nltk_pos_corpus(
            corpus_name="treebank",
            corpus_loader=lambda: treebank.tagged_sents(),
            pos_system="treebank",
        )
        return ds, lambda x: x["tokens"]

    if name == "nps_chat":
        from nltk.corpus import nps_chat
        ds = load_nltk_pos_corpus(
            corpus_name="nps_chat",
            corpus_loader=lambda: nps_chat.tagged_posts(),
            pos_system="nps_chat",
        )
        return ds, lambda x: x["tokens"]
    
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
        if "cath_tags" in example and split_into_words:
            word_ids = enc.word_ids()
            cath_labels = example["cath_tags"] 
            part_labels = example["part_tags"]
            aligned_cath = []
            aligned_part = []
            for wid in word_ids:
                if wid is None:
                    aligned_cath.append("pad")  # OTHER
                    aligned_part.append("pad")  # OTHER
                else:
                    aligned_cath.append(str(cath_labels[wid]))
                    aligned_part.append(str(part_labels[wid]))
                    
            out_dict["cath_tags"] = aligned_cath  # List[int]
            out_dict["part_tags"] = aligned_part  # List[int]

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

    return ds.map(
        _tokenize_and_encode,
        remove_columns=ds.column_names,
        batched=False)


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
    
    if bool(UNKNOWN_POS):
        unknown_list = sorted(list(UNKNOWN_POS))
        raise ValueError(f"Unknown POS tags encountered: {unknown_list}")
    
    ds = subset_and_shuffle(full_ds, subset, shuffle=shuffle)

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    encoder = freeze_encoder(AutoModel.from_pretrained(tokenizer_name))

    ds = encode_examples(name, ds, tok, encoder, text_fn, max_length)
    return ds


def get_dataset(
    tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    name: str = "wikiann",
    split: str = "train",
    dataset_config: dict | None = None,
    subset: int | float | None = None,
    rebuild: bool = False,
    shuffle: bool = False,
    logger: Optional[logging.Logger] = None,
    max_length: int = 512,
    raw_dataset_path: Optional[str] = None,
) -> Dataset:
    name = canonical_name(name)
    path = dataset_path(name, split, dataset_config=dataset_config)
    if dataset_config is None:
        dataset_config = DATASETS_DEFAULT_CONFIG[name]

    if path.exists() and not rebuild:
        if logger is not None:
            logger.info("Loading cached dataset from %s", path)
        ds = load_from_disk(path)
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

def collate(batch, system: str | None = None) -> dict[str, torch.Tensor | list[list[str]]]:
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

    PAD_TAG = "<PAD_TAG>"

    def pad_str_lists(list_of_lists, pad_value):
        max_len = max(len(x) for x in list_of_lists)
        return [
            x + [pad_value] * (max_len - len(x))
            for x in list_of_lists
        ]

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

    if "ner_tags" in batch[0]:
        ner_tags = pad_sequence(
            [ _to_tensor(item["ner_tags"], torch.long) for item in batch ],
            batch_first=True,
            padding_value=-100,  # ignore_index
        )

    if "part_tags" in batch[0]:
        if system is None:
            raise ValueError("POS-tagged dataset but no pos_system provided by dataloader")

        local_map = PART_TO_ID_BY_SYSTEM[system]

        raw_part = [item["part_tags"] for item in batch]
        padded_part = pad_str_lists(raw_part, pad_value="pad")

        part_tags = torch.tensor(
            [[local_map.get(tag, local_map["pad"]) for tag in seq]
            for seq in padded_part],
            dtype=torch.long,
        )

    if "cath_tags" in batch[0]:
        raw_cath = [item["cath_tags"] for item in batch]
        padded_cath = pad_str_lists(raw_cath, pad_value="pad")

        cath_tags = torch.tensor(
            [[CATH_TO_ID.get(tag, CATH_TO_ID["pad"]) for tag in seq]
             for seq in padded_cath],
            dtype=torch.long,
        )

    tokens = [item["tokens"] for item in batch]  # list[list[str]]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "embeddings": embeddings,
        "tokens": tokens,
        "ner_tags": ner_tags if "ner_tags" in batch[0] else None,
        "cath_tags": cath_tags if "cath_tags" in batch[0] else None,
        "part_tags": part_tags if "part_tags" in batch[0] else None,
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
        
        name = canonical_name(split_cfg.dataset)
        system = name if name in USES_SYSTEM else None

        def collate_with_system(batch):
            return collate(batch, system=system)

        return DataLoader(
            ds,
            batch_size=split_cfg.batch_size,
            collate_fn=collate_with_system,
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
