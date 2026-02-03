#!/usr/bin/env python3
"""
Convert Jupyter .ipynb notebooks to .py scripts.

- Preserves cell order.
- Converts markdown cells to commented blocks with a cell header.
- Writes output .py next to the source .ipynb by default.
- Skips .ipynb files inside .ipynb_checkpoints.

Usage:
  python scripts/nb_to_py.py [PATH ...]

PATH can be a directory (searched recursively), a single .ipynb file,
or a shell glob pattern (expanded by your shell).
If omitted, the current working directory is used.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import argparse
import glob as _glob
from datetime import datetime


def iter_notebooks_from_dir(root: Path):
    for p in root.rglob("*.ipynb"):
        if "/.ipynb_checkpoints/" in str(p):
            continue
        yield p


def normalize_inputs(paths: list[str]) -> list[Path]:
    # Expand globs and return explicit Paths (files or dirs)
    if not paths:
        return [Path.cwd()]
    results: list[Path] = []
    for s in paths:
        expanded = _glob.glob(s)
        if not expanded:
            results.append(Path(s))
        else:
            results.extend(Path(p) for p in expanded)
    return results


def convert_notebook(nb_path: Path) -> Path:
    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    lines: list[str] = []
    rel = nb_path
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    header = [
        "# Generated from notebook: {}".format(rel),
        "# Generated at: {}".format(ts),
        "# NOTE: Markdown cells are converted to comments.",
        "#       IPython magics (e.g., %matplotlib) are kept as-is and may require IPython.",
        "",
    ]
    lines.extend(header)

    for i, cell in enumerate(cells, start=1):
        ctype = cell.get("cell_type")
        src = cell.get("source", [])
        # Normalize to list of lines
        if isinstance(src, str):
            src_lines = src.splitlines()
        else:
            # elements already contain newlines usually
            src_lines = []
            for s in src:
                src_lines.extend(s.splitlines())

        if ctype == "markdown":
            lines.append(f"# %% [markdown] cell {i}")
            for s in src_lines:
                lines.append("# " + s)
            lines.append("")
        elif ctype == "code":
            lines.append(f"# %% [code] cell {i}")
            if src_lines:
                lines.extend(src_lines)
            else:
                lines.append("pass")
            lines.append("")
        else:
            # Unknown cell type: keep as commented block
            lines.append(f"# %% [{ctype}] cell {i}")
            for s in src_lines:
                lines.append("# " + s)
            lines.append("")

    out_path = nb_path.with_suffix(".py")
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Convert .ipynb to .py")
    parser.add_argument("paths", nargs="*", help="Directories, files, or glob patterns")
    args = parser.parse_args(argv[1:])

    inputs = normalize_inputs(args.paths)
    notebooks: list[Path] = []
    for p in inputs:
        p = p.resolve()
        if p.is_file() and p.suffix == ".ipynb":
            notebooks.append(p)
        elif p.is_dir():
            notebooks.extend(iter_notebooks_from_dir(p))
        else:
            # Nonexistent or non-ipynb file; ignore but warn
            print(f"Skip (not found or not .ipynb): {p}")

    # Unique and keep deterministic order
    notebooks = sorted(set(notebooks))

    if not notebooks:
        print("No notebooks found.")
        return 0
    print(f"Found {len(notebooks)} notebooks. Converting…")
    out_files: list[Path] = []
    for nb in notebooks:
        try:
            out = convert_notebook(nb)
            out_files.append(out)
            print(f"OK  {nb} -> {out}")
        except Exception as e:
            print(f"ERR {nb}: {e}")
    print(f"Done. Wrote {len(out_files)} .py files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
