"""Jupyter notebook (.ipynb) to Python source conversion with cell/line bookkeeping.

Notebooks are the dominant vehicle for published biomedical ML code, and the
leakage patterns we look for (fit on full data, then split; test set used for
early stopping) frequently span cells.  We therefore concatenate all code cells
in document order, strip IPython magics/shell escapes, and keep a map from the
concatenated line number back to ``(cell_index, line_in_cell)`` so findings can
be reported per cell.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

_MAGIC_RE = re.compile(r"^\s*(%|!|\?)")
_INLINE_MAGIC_RE = re.compile(r"^\s*%%")


@dataclass
class NotebookSource:
    """Concatenated notebook code plus a line -> (cell, line) map."""

    source: str
    line_map: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    n_code_cells: int = 0

    def locate(self, lineno: int) -> Tuple[int, int]:
        """Return ``(cell_index, line_in_cell)`` for a concatenated line number (1-based)."""
        return self.line_map.get(lineno, (-1, lineno))


def _clean_cell(lines: List[str]) -> List[str]:
    """Blank out magics and shell escapes but keep line count unchanged."""
    out: List[str] = []
    for ln in lines:
        if _MAGIC_RE.match(ln) or _INLINE_MAGIC_RE.match(ln):
            out.append("")  # preserve numbering
        else:
            out.append(ln.rstrip("\n"))
    return out


def notebook_to_source(path: str | Path) -> NotebookSource:
    """Read an ``.ipynb`` file and return concatenated Python source.

    Parameters
    ----------
    path:
        Path to a Jupyter notebook (nbformat 4).

    Returns
    -------
    NotebookSource
        ``source`` is valid-looking Python (magics blanked); ``line_map`` maps the
        1-based line number in ``source`` to ``(cell_index, line_in_cell)`` where
        ``cell_index`` counts *all* cells (markdown included) so it matches what a
        reader sees in Jupyter.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        nb = json.load(fh)
    cells = nb.get("cells", [])
    pieces: List[str] = []
    line_map: Dict[int, Tuple[int, int]] = {}
    cursor = 1
    n_code = 0
    for ci, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        n_code += 1
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        lines = src.split("\n")
        cleaned = _clean_cell(lines)
        for li, _ in enumerate(cleaned, start=1):
            line_map[cursor] = (ci, li)
            cursor += 1
        pieces.append("\n".join(cleaned))
        # one separator line per cell so cells never glue into one statement
        line_map[cursor] = (ci, len(cleaned) + 1)
        cursor += 1
    source = "\n".join(pieces) + "\n"
    return NotebookSource(source=source, line_map=line_map, n_code_cells=n_code)
