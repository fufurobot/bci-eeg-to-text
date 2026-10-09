# bci-eeg-to-text

Training an EEG-to-text decoder with **CTC loss** on the
[Chisco](https://github.com/zhangzihan-is-good/Chisco) Chinese imagined-speech
dataset ([OpenNeuro ds005170](https://doi.org/10.18112/openneuro.ds005170.v1.1.2)).

A participant silently imagines reading a Chinese sentence while 128-channel EEG
is recorded at 1000 Hz; the model must recover the sentence from the EEG alone.

## The hard part: there are no stimulus triggers

This dataset is **output-only**. The marker channel contains a single uniform
reward pulse per trial and *no* stimulus-onset triggers, so the usual
BIDS/MNE event route cannot tell you which sentence was imagined when.

Alignment must therefore be recovered. The evidence and reasoning are written up
in [issue #1](https://github.com/fufurobot/bci-eeg-to-text/issues/1); in short:

> Run `run-NNN` corresponds to `textdataset/split_data_{N}.xlsx`, and the i-th
> reward pulse corresponds to the i-th sentence in that file.

Verified for `sub-01_ses-02_task-imagine_run-010`: **150 reward pulses = 150
sentences**, 10.08 s apart.

Because a wrong mapping trains the model on mislabelled data and looks like a
modelling failure, the pipeline **raises** on any pulse/sentence mismatch rather
than truncating or padding to fit.

Known dataset quirks:

- Runs carry a **global** run index, so `run-08` and `run-010` can both live in
  `ses-02`; mapping keys on the run number, never session order.
- Despite the derivatives listing a `task-read`, **no `task-read` EDF exists** —
  all 223 runs are `task-imagine`.
- The pulse is assumed to mark trial **onset**; see `ExtractConfig.pre_samples`.

## Pipeline

```
EDF run ──► reward pulses ──┐
                            ├──► aligned trials ──► CTC model ──► text
split_data_N.xlsx ──────────┘
```

| Module | Responsibility |
| --- | --- |
| `eeg.py` | EDF reading, reward-pulse detection, segment alignment |
| `textdata.py` | stimulus sentences from `split_data_*.xlsx` |
| `tokenizer.py` | character-level CTC vocabulary |
| `data.py` | trial extraction, caching, batching |
| `model.py` | strided convolutional encoder + CTC head |
| `decode.py` | greedy / prefix beam search, CER, sentence accuracy |
| `train.py` | training and evaluation loops |
| `cli.py` | `prepare`, `inspect`, `train`, `evaluate` |

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env && chmod 600 .env   # then fill in credentials
```

`.env` is git-ignored and must never be committed. `.env.example` is the
credential-free template.

## Usage

```bash
# what is in one run? reports pulses vs sentences and whether they align
uv run bci-eeg-to-text inspect \
  eeg-text-bci-data/bci-eeg/ds005170/sub-01/ses-02/eeg/sub-01_ses-02_task-imagine_run-010_eeg.edf

# extract aligned trials into a cache (raw EDFs are huge; this is done once)
uv run bci-eeg-to-text prepare --out cache/trials --decimate 8

# train
uv run bci-eeg-to-text train --cache cache/trials --out runs/exp1 --epochs 10

# evaluate (--beam-width > 1 enables prefix beam search)
uv run bci-eeg-to-text evaluate --cache cache/trials --checkpoint runs/exp1/last.pt
```

## Tests

```bash
uv run pytest -q -m "not realdata"   # unit tests, no dataset needed
uv run pytest -q                     # includes dataset-backed tests
```

Real-data tests are marked `realdata` and skip automatically when the dataset is
not mounted.

## Design notes

**Temporal downsampling is not optional.** Trials are tens of thousands of
samples while targets are 6–15 characters, and CTC requires
`input_length >= target_length`. The encoder strides the time axis by 128×.

**CTC length bookkeeping is exact.** `input_lengths_to_output` replays the
encoder's own layer arithmetic rather than approximating. Over-reporting lengths
makes CTC consume padding frames as if they were signal — a bug that degrades
results quietly instead of raising.

**Metrics are text metrics.** Character error rate and exact-sentence accuracy
are the numbers that matter; a model can look plausible per character while
getting whole sentences wrong.

## Dataset

Read-only; not vendored here. `eeg-text-bci-data` is a symlink to the dataset
mount and is git-ignored.

- Chisco, Zhang et al. — [GitHub](https://github.com/zhangzihan-is-good/Chisco)
- DOI: [10.18112/openneuro.ds005170.v1.1.2](https://doi.org/10.18112/openneuro.ds005170.v1.1.2)
- License: CC0
