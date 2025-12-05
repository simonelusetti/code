import logging
import os
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from prettytable import PrettyTable
from tqdm import tqdm


def should_disable_tqdm(
) -> bool:
    """Return True when tqdm progress bars should be disabled."""
    override = os.environ.get("DISABLE_TQDM")
    if override is not None:
        return override.strip().lower() not in {"0", "false", "no", "off"}

    try:
        return not sys.stderr.isatty()
    except Exception:
        return True


class TqdmLoggingHandler(logging.Handler):
    def emit(
        self,
        record: logging.LogRecord,
    ) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
            sys.stdout.flush()
        except Exception:  # pragma: no cover - logging fallback
            self.handleError(record)


def get_logger(
    logfile: str = "train.log",
) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if logger.hasHandlers():
        logger.handlers.clear()

    ch = TqdmLoggingHandler()
    ch.setLevel(logging.INFO)
    ch_format = "%(asctime)s - %(levelname)s - %(message)s"
    ch.setFormatter(logging.Formatter(ch_format))

    fh = logging.FileHandler(Path(logfile))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def make_table(
    fields: Sequence[str],
    rows: Iterable[Sequence[object]],
) -> PrettyTable:
    table = PrettyTable()
    table.field_names = fields
    for row in rows:
        table.add_row(row)
    return table


def dict_to_table(
    losses: Mapping[str, float],
) -> PrettyTable:
    fields = losses.keys()
    rows = [[f"{losses.get(k, 0.0):.6f}" for k in fields]]
    return make_table(fields, rows)


def make_slot_table(
    metrics: Sequence[tuple[str, float, float, float, int, int, int, int]],
) -> str:
    """
    Build a PrettyTable from a list of tuples:
    (key, f1, precision, recall, tp, fp, fn, total)
    Returns the table string for logging.
    """
    fields = ["slot", "f1", "precision", "recall", "tp", "fp", "fn", "total"]
    rows = []
    for key, f1, p, r, tp, fp, fn, total in metrics:
        rows.append([key, f"{f1:.4f}", f"{p:.4f}", f"{r:.4f}", int(tp), int(fp), int(fn), int(total)])
    return make_table(fields, rows).get_string()
