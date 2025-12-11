from __future__ import annotations

import logging, gzip, torch, json 

from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import Callable, Optional, Tuple
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from dora import to_absolute_path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoModel, AutoTokenizer
from urllib.request import urlretrieve
from conllu import parse_incr

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

PAD_TAG = "<PAD_TAG>"
POS_TAGSET = {PAD_TAG}

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
) -> Dataset:
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
        sentences.append({
            "tokens": [w for (w, _) in sent],
            "pos": [t for (_, t) in sent],
        })
    return Dataset.from_list(sentences)

            
# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------


def collate_with_map(pos_map: dict) -> Callable:
    from .utils import pad_str_lists, to_tensor
    def collate(batch: list[dict]) -> dict[str, torch.Tensor | list[list[str]]]:
        input_ids = pad_sequence(
            [to_tensor(item["input_ids"], torch.long) for item in batch],
            batch_first=True,
            padding_value=0,
        )
        attention_masks = pad_sequence(
            [to_tensor(item["attention_mask"], torch.long) for item in batch],
            batch_first=True,
            padding_value=0,
        )
        embeddings = pad_sequence(
            [to_tensor(item["embeddings"], torch.float32) for item in batch],
            batch_first=True,
            padding_value=0.0,
        )
        sentence_reps = torch.stack(
            [to_tensor(item["sentence_rep"], torch.float32) for item in batch],
            dim=0,
        )
        labels = None
        if "labels" in batch[0]:
            if isinstance(batch[0]["labels"][0], bool):
                labels = pad_sequence(
                    [to_tensor(item["labels"], torch.long) for item in batch],
                    batch_first=True,
                    padding_value=-100,  # ignore_index
                )
            else:
                raw_pos = [item["labels"] for item in batch]
                padded_pos = pad_str_lists(raw_pos, pad_value=PAD_TAG)
                pad_idx = pos_map[PAD_TAG]
                labels = torch.tensor(
                    [[pos_map.get(tag, pad_idx) for tag in seq]
                    for seq in padded_pos],
                    dtype=torch.long,
                )            

        return {
            "embeddings": embeddings,
            "attention_mask": attention_masks,
            "sentence_reps": sentence_reps,
            "input_ids": input_ids,
            "tokens": [item["tokens"] for item in batch],
            "labels": labels,
        }
        
    return collate


def encode_examples(
    shared_cfg: dict,
    ds: Dataset,
    text_fn: Callable,
) -> Dataset:
    from .utils import sbert_encode
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sbert = SentenceTransformer(shared_cfg["sentence_model_name"])
    tok = AutoTokenizer.from_pretrained(shared_cfg.tokenizer_name, use_fast=True)
    encoder = freeze_encoder(AutoModel.from_pretrained(shared_cfg.tokenizer_name))
    
    def _tokenize_and_encode(example, sbert):
        text = text_fn(example)
        
        enc = tok(text,truncation=True,max_length=shared_cfg["max_length"],
            is_split_into_words=isinstance(text, (list, tuple)))
        
        input_ids = torch.tensor(enc["input_ids"], device=device).unsqueeze(0)
        attention_mask = torch.tensor(enc["attention_mask"], device=device).unsqueeze(0)
        with torch.no_grad():
            out = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=False,
                return_dict=True,
            )

        labels_name = {"pos", "ner"} & example.keys()
        aligned_labels = []
        if labels_name:
            word_ids = enc.word_ids()
            labels_name = labels_name.pop()
            labels = example[labels_name]
            if labels_name == "pos":
                POS_TAGSET.update(labels)
            for wid in word_ids:
                if wid is None and labels_name == "pos":
                    aligned_labels.append(PAD_TAG)  
                elif wid is None and labels_name == "ner":
                    aligned_labels.append(False)
                elif labels_name == "pos":
                    aligned_labels.append(str(labels[wid]))
                elif labels_name == "ner":
                    aligned_labels.append(labels[wid] == 'O')

        return {
            "input_ids": enc["input_ids"],                           
            "attention_mask": enc["attention_mask"],                 
            "embeddings": out.last_hidden_state.squeeze(0).cpu().to(torch.float32).tolist(),                      
            "tokens": tok.convert_ids_to_tokens(enc["input_ids"]),
            "sentence_rep": sbert_encode(sbert, out.last_hidden_state, attention_mask).detach(),
            "labels": aligned_labels if labels_name else None,
        }

    return ds.map(
        lambda example: _tokenize_and_encode(example, sbert),
        remove_columns=ds.column_names,
        batched=False)


def resolve_dataset(
    cfg: dict,
    raw_dataset_path: Optional[str],
) -> Tuple[Dataset, Callable]:
    """
    Returns (dataset, text_fn) where text_fn(example) -> tokens or text.
    """
    name = canonical_name(cfg["dataset"])
    ds = None
    text_fn = None
    
    if raw_dataset_path is not None:
        raw_root = Path(raw_dataset_path)
        raw_split_path = raw_root / cfg["split"]
        if not raw_split_path.exists():
            raise FileNotFoundError(f"Raw dataset split not found at {raw_split_path}")
        ds = load_from_disk(str(raw_split_path))

    if name == "cnn_dailymail":
        if ds is None:
            ds = load_dataset("cnn_dailymail", cfg["config"]["version"], split=cfg["config"]["split"])
        return ds, lambda x: x[cfg["config"]["field"]]

    if name == "wikiann":
        if ds is None:
            ds = load_dataset("wikiann", cfg["config"]["language"], split=cfg["config"]["split"])
        return ds, lambda x: x["tokens"]

    if name == "conll2003":
        if ds is None:
            ds = load_dataset("conll2003", split=cfg["config"]["split"])
        return ds, lambda x: x["tokens"]
    
    if name == "parasci":
        ds = load_from_disk(to_absolute_path("./data/cache/parasci"))
        return ds[cfg["split"]], lambda x: x["text"]

    if name == "parasci-concat":
        ds = load_from_disk(to_absolute_path("./data/cache/parasci-concat"))
        return ds[cfg["split"]], lambda x: x["text"]
    
    if name == "ud":
        text_fn = lambda x: x["tokens"]
        if ds is not None:
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
                    sentences.append({
                        "tokens": [tok["form"] for tok in sent],
                        "pos": [tok["upos"] for tok in sent]
                    })
            return sentences
        datasets_dict = {sp: load_conllu(path) for sp, path in split_paths.items()}
        ds = DatasetDict({sp: Dataset.from_list(data) for sp, data in datasets_dict.items()})
        return ds[cfg["split"]], text_fn
    
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
        gz_path = raw_root / files[cfg["split"]]
        txt_path = raw_root / files[cfg["split"]].replace(".gz", "")
        if not gz_path.exists():
            urlretrieve(base_url + files[cfg["split"]], gz_path)
        if not txt_path.exists():
            with gzip.open(gz_path, "rt", encoding="utf-8") as f_in, \
                 open(txt_path, "w", encoding="utf-8") as f_out:
                for line in f_in:
                    f_out.write(line)
        # Parse
        def _format_sentence(tokens, pos_tags):
            return{
                "tokens": tokens,
                "pos": pos_tags,
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
        if ds is None:
            from nltk.corpus import brown
            ds = load_nltk_pos_corpus(
                corpus_name="brown",
                corpus_loader=lambda: brown.tagged_sents(tagset="universal"),
            )
        return ds, lambda x: x["tokens"]

    if name == "treebank":
        if ds is None:
            from nltk.corpus import treebank
            ds = load_nltk_pos_corpus(
                corpus_name="treebank",
                corpus_loader=lambda: treebank.tagged_sents(),
            )
        return ds, lambda x: x["tokens"]

    if name == "nps_chat":
        if ds is None:
            from nltk.corpus import nps_chat
            ds = load_nltk_pos_corpus(
                corpus_name="nps_chat",
                corpus_loader=lambda: nps_chat.tagged_posts(),
            )
        return ds, lambda x: x["tokens"]
    
    raise ValueError(f"Unknown dataset name: {name}")


def build_dataset(
    shared_cfg: dict,
    split_cfg: dict,
    raw_dataset_path: Optional[str],
) -> Dataset:
    full_ds, text_fn = resolve_dataset(split_cfg, raw_dataset_path)
    ds = subset_and_shuffle(full_ds, split_cfg.subset, shared_cfg.shuffle)
    ds = encode_examples(shared_cfg,ds,text_fn)
    return ds


def get_dataset(
    shared_cfg: dict,
    split_cfg: dict,
    logger: Optional[logging.Logger] = None,
    raw_dataset_path: Optional[str] = None,
) -> Dataset:
    name = canonical_name(split_cfg.dataset)
    path = dataset_path(name, split_cfg.split, dataset_config=split_cfg.get("config", None))
    if split_cfg["config"] is None:
        split_cfg["config"] = DATASETS_DEFAULT_CONFIG[name]

    if path.exists() and not shared_cfg["rebuild"]:
        if logger is not None:
            logger.info("Loading cached dataset from %s", path)
        ds = load_from_disk(path)
        ds = subset_and_shuffle(ds, split_cfg.subset, shuffle=shared_cfg.shuffle)
        return ds

    if logger is not None:
        logger.info("Building dataset for %s/%s (subset=%s)", name, split_cfg.split, split_cfg.subset)
        
    ds = build_dataset(
        shared_cfg=shared_cfg,
        split_cfg=split_cfg,
        raw_dataset_path=raw_dataset_path,
    )
    
    if len(POS_TAGSET) > 1:
        pos_map = {value: i for i, value in enumerate(sorted(POS_TAGSET))}
        with open(to_absolute_path(path.parent / f"{name}_pos_map.json"), "w") as f:
            json.dump(pos_map, f)

    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(path)
    return ds


def build_loader(
    shared_cfg: dict,
    split_cfg: dict,
    pos_map: Optional[dict],
    device: str,
    logger: Optional[logging.Logger],
) -> DataLoader:
    ds = get_dataset(shared_cfg,split_cfg,logger,None)
    return DataLoader(
        ds,
        batch_size=shared_cfg.batch_size,
        collate_fn=collate_with_map(pos_map),
        num_workers=shared_cfg.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(shared_cfg.num_workers > 0),
        shuffle=shared_cfg.shuffle,
    )


def initialize_dataloaders(
    cfg: dict,
    logger: Optional[logging.Logger] = None,
    device: str = "cpu",
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Optional[dict]]:
    
    name = canonical_name(cfg.train.dataset)
    path = dataset_path(name, cfg.train.split, dataset_config=cfg.train.get("config", None))

    try:
        with open(to_absolute_path(path.parent / f"{name}_pos_map.json"), "r") as f:
            pos_map = json.load(f)
    except FileNotFoundError:
        pos_map = None

    train_dl = build_loader(cfg.shared,cfg.train,pos_map,device,logger)
    
    eval_dl = build_loader(cfg.shared,cfg.eval,pos_map,device,logger)
  
    if "dev" in cfg and cfg["dev"]["dataset"] is not None:
        dev_dl = build_loader(cfg.shared,cfg.dev,pos_map,device,logger)
        full_eval_ds = ConcatDataset([dev_dl.dataset, eval_dl.dataset])
        eval_dl = DataLoader(
            full_eval_ds,
            batch_size=cfg.eval.batch_size,
            collate_fn=collate_with_map(pos_map),
            num_workers=cfg.eval.num_workers,
            pin_memory=(device == "cuda"),
            persistent_workers=(cfg.eval.num_workers > 0),
            shuffle=cfg.eval.shuffle,
        )

    return train_dl, eval_dl, pos_map
