import copy
import os
from typing import Any, Callable, Optional, Sequence, Tuple

import torch
from torch import nn, Tensor
from sentence_transformers import SentenceTransformer


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
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    if "ner_tags" in batch:
        return embeddings, attention_mask, input_ids, batch["ner_tags"].to(device, non_blocking=True)
    return embeddings, attention_mask, input_ids, None


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
    pooler: torch.nn.Module,
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    features = {"token_embeddings": embeddings, "attention_mask": attention_mask}
    pooled = pooler(features)
    return pooled["sentence_embedding"] if isinstance(pooled, dict) else pooled

def freeze_encoder(
    encoder: AutoModel,
) -> AutoModel:
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder