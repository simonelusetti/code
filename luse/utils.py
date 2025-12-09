from __future__ import annotations

import copy
import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel

from .data import CATH_TO_ID, PART_TO_ID_BY_SYSTEM, canonical_name

def resolve_aggregate_fn(
    name: str,
) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
    if name == "sum":
        return None
    if name == "prod":
        return aggregate_product
    raise ValueError(f"Unknown recon aggregation '{name}'.")


def aggregate_product(
    subsent_repr: torch.Tensor,
) -> torch.Tensor:
    return subsent_repr.prod(dim=1)


def format_gold_spans(
    ids: Sequence[int],
    tokens: Sequence[str],
    gold_labels: Sequence[int],
    tokenizer: Any,
) -> str:
    buf = ""
    buf_labels = []
    words, word_labels = [], []
    for tok_id, tok_str, lab in zip(ids, tokens, gold_labels):
        if tok_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
            continue
        if tok_str.startswith("##"):
            buf += tok_str[2:]
            buf_labels.append(lab)
        else:
            if buf:
                words.append(buf)
                word_labels.append(1 if any(l != 0 for l in buf_labels) else 0)
            buf = tok_str
            buf_labels = [lab]
    if buf:
        words.append(buf)
        word_labels.append(1 if any(l != 0 for l in buf_labels) else 0)
    out, span = [], []
    for w, l in zip(words, word_labels):
        if l:
            span.append(w)
        else:
            if span:
                out.append(f"[[{' '.join(span)}]]")
                span = []
            out.append(w)
    if span:
        out.append(f"[[{' '.join(span)}]]")
    return " ".join(out)


def merge_subwords(
    ids: Sequence[int],
    tokens: Sequence[str],
    tokenizer: Any,
) -> list[str]:
    buf = ""
    words = []

    def flush(acc):
        if acc:
            words.append(acc)
        return ""

    for tok_id, tok_str in zip(ids, tokens):
        if tok_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
            continue
        if tok_str.startswith("##"):
            buf += tok_str[2:]
        else:
            buf = flush(buf)
            buf = tok_str

    buf = flush(buf)
    return words


def merge_spans(
    ids: Sequence[int],
    tokens: Sequence[str],
    gates: Sequence[float] | torch.Tensor,
    tokenizer: Any,
    thresh: float = 0.5,
) -> str:
    buf, buf_gs = "", []
    words, word_gates = [], []

    def flush(acc, gs):
        if acc:
            words.append(acc)
            word_gates.append(sum(gs) / len(gs))
        return "", []

    for tok_id, tok_str, g in zip(ids, tokens, gates):
        if tok_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
            continue
        if tok_str.startswith("##"):
            buf += tok_str[2:]
            buf_gs.append(g)
        else:
            buf, buf_gs = flush(buf, buf_gs)
            buf, buf_gs = tok_str, [g]

    buf, buf_gs = flush(buf, buf_gs)

    out_tokens, span_buf = [], []

    def flush_span(span_buf):
        if span_buf:
            out_tokens.append(f"[[{' '.join(span_buf)}]]")
        return []

    for word, g in zip(words, word_gates):
        if g >= thresh:
            span_buf.append(word)
        else:
            span_buf = flush_span(span_buf)
            out_tokens.append(word)

    span_buf = flush_span(span_buf)
    return " ".join(out_tokens)


def counts(
    pred_mask: torch.Tensor,
    gold_mask: torch.Tensor,
) -> Tuple[int, int, int]:
    tp = (pred_mask & gold_mask).sum().item()
    fp = (pred_mask & (~gold_mask)).sum().item()
    fn = ((~pred_mask) & gold_mask).sum().item()
    return tp, fp, fn


def metrics_from_counts(
    tp: int,
    fp: int,
    fn: int,
) -> Tuple[float, float, float]:
    tp = float(tp)
    fp = float(fp)
    fn = float(fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return f1, precision, recall


def configure_runtime(
    cfg: Any,
) -> None:
    runtime = cfg.runtime
    num_threads = runtime.num_threads
    os.environ["TOKEN_PARALLELISM"] = str(runtime.token_parallelism)
    if not num_threads:
        return
    try:
        num_threads = int(num_threads)
    except (TypeError, ValueError):
        return
    if num_threads <= 0:
        return
    value = str(num_threads)
    for env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[env_var] = value
    try:
        import torch

        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(max(1, num_threads // 2))
    except Exception:
        pass


def ensure_finite(
    node: Any,
    name: str,
    tensor: torch.Tensor,
) -> None:
    if torch.isfinite(tensor).all():
        return
    node.logger.warning(f"Non-finite values detected in {name} at node {node.path}")
    finite_mask = torch.isfinite(tensor)
    if finite_mask.any():
        finite_vals = tensor[finite_mask]
        stats = (
            finite_vals.min().item(),
            finite_vals.max().item(),
            finite_vals.mean().item(),
        )
    else:
        stats = (float("nan"), float("nan"), float("nan"))
    bad_idx = (~finite_mask).nonzero(as_tuple=False)[:5].tolist()
    raise RuntimeError(
        f"Non-finite {name} at node {node.path}; stats min/max/mean={stats}; sample bad idx={bad_idx}"
    )


def prepare_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    embeddings = batch["embeddings"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    extra = {}
    if batch["ner_tags"] is not None:
        extra["ner_tags"] = batch["ner_tags"]
    if batch["cath_tags"] is not None:
        extra["cath_tags"] = batch["cath_tags"]
    if batch["part_tags"] is not None:
        extra["part_tags"] = batch["part_tags"]
    return embeddings, attention_mask, batch["tokens"], extra


def load_sbert(
    model_name: str,
    device: torch.device | None = None,
) -> Tuple[torch.nn.Module, int]:
    base = SentenceTransformer(model_name)
    encoder = base[0]
    if device is not None:
        encoder = encoder.to(device)
    pooler = copy.deepcopy(base[1]).to(device) if device is not None else copy.deepcopy(base[1])
    hidden_dim = encoder.auto_model.config.hidden_size
    return pooler, hidden_dim


def sbert_encode(
    sbert: SentenceTransformer,
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    features = {
        "token_embeddings": embeddings,
        "attention_mask": attention_mask,
    }
    pooled = sbert[1](features)             # Pooling layer
    pooled = sbert[2](pooled)               # Normalize layer
    return pooled["sentence_embedding"] # Final embedding [B, H]

def freeze_encoder(
    encoder: AutoModel,
) -> AutoModel:
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder


def wp_to_text(tokens: list[str]) -> str:
    """Reconstruct text from BERT wordpieces."""
    words = []
    current = ""
    for tok in tokens:
        if tok.startswith("##"):
            current += tok[2:]
        else:
            if current:
                words.append(current)
            current = tok
    if current:
        words.append(current)
    return " ".join(words)


def sbert_encode_texts(sbert, texts, device):
    """
    texts: list[str]
    returns: torch.Tensor on device with shape [B, hidden_dim]
    """
    with torch.no_grad():
        reps = sbert.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    return reps.to(device)


def format_dict(d, new_liners=None):
    extra_newline_after = new_liners or set()
    lines = []

    for k, v in d.items():
        if isinstance(v, (int, float)):
            v = f"{v:.5f}"
        lines.append(f"{k}: {v}")
        if k in extra_newline_after:
            lines.append("") 

    return "\n".join(lines)


class Counts(dict):
    def __init__(
        self, 
        classes: list,
        pad: int,
        classes_str: dict | None = None,
        classes_mask: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.data = {}
        self.pad = pad
        self.classes = classes
        self.classes_str = classes_str
        if classes_mask is None:
            for c in self.classes:
                if c != self.pad:
                    self.data[c] = 0
        else:
            for c in self.classes:
                if c != self.pad:
                    self.data[c] = ((classes_mask == c) & mask).sum().item()

    def __add__(self, other: Counts) -> Counts:
        if not isinstance(other, Counts):
            raise ValueError("Can only add Counts to Counts.")

        result = Counts(self.classes, self.pad, classes_str=self.classes_str)
        for c in self.classes:
            if c != self.pad:
                result.data[c] = self.data[c] + other.data[c]
        return result
    
    def __truediv__(self, other: Counts) -> Counts:
        if not isinstance(other, Counts):
            raise ValueError("Can only divide Counts by Counts.")

        result = Counts(self.classes, self.pad, classes_str=self.classes_str)
        for c in self.classes:
            if c != self.pad:
                if other.data[c] == 0:
                    result.data[c] = 0.0
                else:
                    result.data[c] = self.data[c] / other.data[c]
        return result
    
    def __str__(self) -> str:
        data_dict = {
            **{f"{k}": v for k, v in self.data.items()},
        }
        sorted_dict = dict(sorted(data_dict.items(), key=lambda x: x[1], reverse=True))
        if self.classes_str is not None:
            sorted_dict = {self.classes_str[int(id_)]: value for id_, value in sorted_dict.items()}
        return format_dict(sorted_dict)
    
    def preferences_over_total(self, total: int) -> Counts:
        result = Counts(self.classes, self.pad, classes_str=self.classes_str)
        for c in self.classes:
            if c != self.pad:
                result.data[c] = self.data[c] / total if total > 0 else 0.0
        return result
    
    def preferences(self) -> Counts:
        result = Counts(self.classes, self.pad, classes_str=self.classes_str)
        total = sum(self.data[c] for c in self.classes if c != self.pad)
        for c in self.classes:
            if c != self.pad:
                result.data[c] = self.data[c] / total if total > 0 else 0.0
        return result
    
def new_counts(
        dataset_name: str,
        gates: torch.Tensor,
        attention_mask: torch.Tensor,
        extra: Dict,
        local_part_to_id: Dict,
        device: torch.device,
        cath_str: dict, 
        part_str: dict,
    ) -> Tuple[Counts, Counts, Counts, Counts, int]:
        PAD_CATH = CATH_TO_ID["pad"]
        PAD_PART = local_part_to_id["pad"]
        local_part_to_id = PART_TO_ID_BY_SYSTEM[
            canonical_name(dataset_name)
        ]

        # flatten
        attn = attention_mask.bool().view(-1)
        preds = (gates > 0.5).bool().view(-1)

        # integer tensors, shape (B, L)
        cath_tags = extra["cath_tags"].to(device)
        part_tags = extra["part_tags"].to(device)

        flat_cath = cath_tags.view(-1)
        flat_part = part_tags.view(-1)

        new_gold_cath = Counts(CATH_TO_ID.values(), PAD_CATH, classes_mask=flat_cath, mask=attn, classes_str=cath_str)
        new_pred_cath = Counts(CATH_TO_ID.values(), PAD_CATH, classes_mask=flat_cath, mask=attn & preds, classes_str=cath_str)

        new_gold_part = Counts(local_part_to_id.values(), PAD_PART, classes_mask=flat_part, mask=attn, classes_str=part_str)
        new_pred_part = Counts(local_part_to_id.values(), PAD_PART, classes_mask=flat_part, mask=attn & preds, classes_str=part_str)
        
        return new_pred_cath, new_gold_cath, new_pred_part, new_gold_part, preds.sum().item()