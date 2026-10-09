#!/usr/bin/env python
"""Verify the reward-pulse / sentence alignment contract across the dataset.

The claim under test (issue #1): run ``run-NNN`` pairs with
``textdataset/split_data_N.xlsx``, and the i-th reward pulse corresponds to the
i-th sentence.

Only the single ``Trigger`` channel is read, so this is fast even on a slow
mount: roughly one second per run against the full 128-channel read's minute.

Usage:
    uv run python scripts/verify_alignment.py [--all] [--out FILE]

By default this samples one run per subject per session. Pass ``--all`` for the
full sweep (223 runs), which is only practical on a fast local disk.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyedflib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bci_eeg_to_text.eeg import REWARD_PULSE, is_annex_pointer  # noqa: E402

DEFAULT_ROOT = Path("eeg-text-bci-data/bci-eeg/ds005170")
_FILENAME_RE = re.compile(r"(sub-\d+)_ses-(\d+)_task-(\w+)_run-(\d+)_eeg\.edf")


def count_sentences(path: Path) -> int:
    """Number of data rows in a ``split_data_N.xlsx`` workbook."""
    with zipfile.ZipFile(path) as archive:
        sheet = archive.read("xl/worksheets/sheet1.xml").decode("utf-8")
    return len(re.findall(r"<row[^>]*>", sheet)) - 1


def check_run(args: tuple[str, str]) -> dict[str, object]:
    edf_path, text_root = args
    name = os.path.basename(edf_path)
    match = _FILENAME_RE.match(name)
    if not match:
        return {"file": name, "error": "unrecognised filename"}
    subject, session, task, run = (
        match.group(1),
        int(match.group(2)),
        match.group(3),
        int(match.group(4)),
    )
    record: dict[str, object] = {
        "subject": subject,
        "session": session,
        "task": task,
        "run": run,
    }

    if is_annex_pointer(edf_path):
        record["status"] = "missing"
        return record

    try:
        with pyedflib.EdfReader(edf_path) as reader:
            trigger = reader.readSignal(reader.getSignalLabels().index("Trigger"))
    except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
        record["status"] = "error"
        record["error"] = str(exc)
        return record

    # Count pulse onsets: a rising edge into the reward value.
    is_pulse = trigger == REWARD_PULSE
    rising = is_pulse & ~np.concatenate(([False], is_pulse[:-1]))
    n_pulses = int(rising.sum())

    split_path = Path(text_root) / f"split_data_{run}.xlsx"
    if not split_path.exists():
        record["status"] = "error"
        record["error"] = f"missing {split_path.name}"
        return record
    n_sentences = count_sentences(split_path)

    record["pulses"] = n_pulses
    record["sentences"] = n_sentences
    record["status"] = "aligned" if n_pulses == n_sentences else "mismatch"
    return record


def select_runs(root: Path, use_all: bool) -> list[Path]:
    runs = sorted(glob.glob(str(root / "sub-*/ses-*/eeg/*_eeg.edf")))
    if use_all:
        return [Path(r) for r in runs]
    # One run per (subject, session) keeps the sweep representative but cheap.
    seen: set[tuple[str, str]] = set()
    picked: list[Path] = []
    for run in runs:
        match = _FILENAME_RE.match(os.path.basename(run))
        if not match:
            continue
        key = (match.group(1), match.group(2))
        if key not in seen:
            seen.add(key)
            picked.append(Path(run))
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--all", action="store_true", help="check every run")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None, help="write JSON results here")
    args = parser.parse_args()

    if not args.root.exists():
        print(f"dataset root not found: {args.root}", file=sys.stderr)
        return 2

    runs = select_runs(args.root, args.all)
    text_root = str(args.root / "textdataset")
    print(f"checking {len(runs)} runs ({'all' if args.all else 'sampled'})", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(check_run, [(str(r), text_root) for r in runs]))

    counts: dict[str, int] = {}
    for record in results:
        counts[record["status"]] = counts.get(record["status"], 0) + 1

    print("\nresults:", json.dumps(counts, indent=2, sort_keys=True))
    for record in results:
        if record["status"] in {"mismatch", "error"}:
            print(
                f"  {record['status'].upper()}: {record['subject']} "
                f"ses-{record['session']} run-{record['run']} "
                f"{record.get('pulses')} vs {record.get('sentences')} "
                f"{record.get('error', '')}"
            )

    aligned = counts.get("aligned", 0)
    checked = aligned + counts.get("mismatch", 0)
    print(f"\naligned {aligned}/{checked} runs that contain real data")
    if counts.get("missing"):
        print(f"{counts['missing']} runs are unfetched git-annex pointers (skipped)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"results": results, "counts": counts}, indent=2), encoding="utf-8"
        )
        print(f"wrote {args.out}")

    # Mismatches are a hard failure: they invalidate the alignment rule.
    return 1 if counts.get("mismatch") else 0


if __name__ == "__main__":
    raise SystemExit(main())
