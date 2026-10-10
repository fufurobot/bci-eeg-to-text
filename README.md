# bci-eeg-to-text

Training an EEG-to-text decoder with **CTC loss** over EEG-text datasets.

The architecture is deliberately dataset-agnostic: a dataset is only a source of
`(EEG segment, text)` pairs, and the tokenizer, model, CTC loss, decoder and
metrics are identical regardless of where those pairs came from. Two corpora are
wired up:

| dataset | task | alignment | subjects | usable runs |
| --- | --- | --- | --- | --- |
| [ds004952](https://doi.org/10.18112/openneuro.ds004952.v1.2.2) ChineseEEG | silent reading | **stimulus markers** (`ROWS`/`ROWE`) | 10 (5F/5M) | 221 |
| [ds005170](https://doi.org/10.18112/openneuro.ds005170.v1.1.2) Chisco | imagined speech | recovered from reward pulses | 4 (2F/2M) | 168 |

Both are 128-channel EEG. `--dataset merged` trains on both at once.

## Alignment is the hard part, and it differs per dataset

**ChineseEEG (ds004952)** ships real triggers, so the pairing is *read*:

```
start = ROWS_k.sample          stop = next ROWE sample
text  = k-th data row of segmented_Chinense_novel_run_N.xlsx
```

A row's duration equals its highlighted character count times 0.35 s, which makes
the mapping checkable. Two non-obvious facts, both verified:

- **Chapter-banner rows must be kept.** They are rendered on screen and occupy a
  real interval. Keeping them gives 383 rows for 383 intervals with a
  duration/length correlation of +0.895; dropping them collapses it to +0.003.
- **Matching counts is not sufficient.** `ses-LittlePrince run-01` has an extra
  `ROWE` and an orphaned leading `ROWE`, so its durations are *anti*-correlated
  with text length (-0.135 against ~+0.88 for healthy runs). The loader therefore
  also requires correlation ≥ 0.5, which rejects exactly those 8 runs.

126 of 168 `.vhdr` headers also contain an upstream typo (`GranettDream` for
`GarnettDream`). They are repaired through a scratch mirror, since the dataset is
read-only.

**Chisco (ds005170)** is output-only: one uniform reward pulse per trial and *no*
stimulus-onset markers, so alignment is *recovered* — run `run-NNN` pairs with
`split_data_N.xlsx`, and the i-th pulse with the i-th sentence
([issue #1](https://github.com/fufurobot/bci-eeg-to-text/issues/1)). Verified:
150 pulses = 150 sentences for `sub-01_ses-02_run-010`.

Because a wrong pairing trains on mislabelled data and looks like a modelling
failure rather than a bug, **every loader validates and raises** instead of
truncating or padding to fit.

## Splits are participant-level and sex-balanced

Splitting by run leaks speaker identity and inflates scores, so the default is by
**subject**, with sex balanced across parts:

```bash
uv run bci-eeg-to-text split --dataset chisco       # 1:1  -> 2 train, 2 test (1F/1M each)
uv run bci-eeg-to-text split --dataset chineseeeg   # 3:1:1 -> 6/2/2 (3F+3M, 1F+1M, 1F+1M)
uv run bci-eeg-to-text split --dataset merged       # 14 subjects, namespaced ids
```

Run-level splitting is available for within-participant experiments but must be
requested explicitly, prints a WARNING, and is recorded in the output JSON:

```bash
uv run bci-eeg-to-text split --dataset chisco --level run
```

The two datasets both use bare `sub-NN` ids, so merged mode namespaces them
(`chisco:sub-01`, `chineseeeg:sub-04`); without that they would collide.

## Dataset notes

- The 46 git-annex pointers in ds005170 are **deliberate** — the drive is too
  small to hold everything — so they are skipped quietly rather than reported as
  errors, and ds005170 genuinely has 4 usable subjects
  ([issue #2](https://github.com/fufurobot/bci-eeg-to-text/issues/2)).
- Runs carry a **global** run index, so `run-08` and `run-010` both live in
  `ses-02`; mapping keys on the run number, never session order.
- No `task-read` EDF exists in ds005170 despite the derivatives listing one.
- The ds005170 pulse is assumed to mark trial **onset**; see
  `ChiscoSource.pre_samples`.
## Pipeline

```
EDF run ──► reward pulses ──┐
                            ├──► aligned trials ──► CTC model ──► text
split_data_N.xlsx ──────────┘
```

| Module | Responsibility |
| --- | --- |
| `backbones.py` | interchangeable encoders (`scratch`, `bert-ctc`, NeuroLM modes) |
| `neurolm.py` | loading NeuroLM components and checkpoints |
| `neurolm_native.py` | native NeuroLM tokenizer (patch embed, transformer, VQ) |
| `sources.py` | `DatasetSource` protocol, registry, `participants.tsv` parsing |
| `chineseeeg.py` | ds004952 reader: events, row intervals, novel text, header repair |
| `chisco.py` | ds005170 reader: reward-pulse recovery |
| `splits.py` | participant-level, sex-balanced train/valid/test partitioning |
| `eeg.py` | EDF reading, pulse detection, segment alignment |
| `textdata.py` | ds005170 stimulus sentences from `split_data_*.xlsx` |
| `tokenizer.py` | character-level CTC vocabulary |
| `data.py` | trial caching and batching over either source |
| `model.py` | strided convolutional encoder + CTC head |
| `decode.py` | greedy / prefix beam search, CER, sentence accuracy |
| `train.py` | training and evaluation loops |
| `cli.py` | `prepare`, `split`, `inspect`, `train`, `evaluate` |

## Training modes

The encoder is interchangeable; the CTC loss, decoder and metrics are shared.
`--mode` selects one:

| mode | encoder | typical use |
| --- | --- | --- |
| `scratch` | strided conv encoder | baseline, no downloads |
| `bert-ctc` | pretrained HuggingFace BERT + CTC head | reuse language representations |
| `neurolm-sft` | full [NeuroLM](https://github.com/935963004/NeuroLM) model | supervised fine-tuning |
| `neurolm-adapter` | HuggingFace causal LM + adapter | bridge a NeuroProcessor to an LM |

```bash
# baseline
uv run bci-eeg-to-text train --cache cache/trials --out runs/scratch --mode scratch

# pretrained BERT with the backbone frozen: only the adapter and head train
uv run bci-eeg-to-text train --cache cache/trials --out runs/bert \
  --mode bert-ctc --pretrained bert-base-chinese --freeze-backbone

# fine-tune the whole NeuroLM model
uv run bci-eeg-to-text train --cache cache/trials --out runs/neurolm-sft \
  --mode neurolm-sft --lm-name gpt2

# causal LM + adapter, training our own NeuroProcessor from scratch
uv run bci-eeg-to-text train --cache cache/trials --out runs/adapter \
  --mode neurolm-adapter --lm-name gpt2 --processor scratch

# same, but reusing a pretrained NeuroProcessor (NeuroLM's own recipe)
uv run bci-eeg-to-text train --cache cache/trials --out runs/adapter-pre \
  --mode neurolm-adapter --lm-name gpt2 --processor pretrained \
  --pretrained checkpoints/VQ.pt
```

### Why these work the way they do

**EEG has no token ids.** For `bert-ctc` the signal is projected into BERT's
hidden space and passed as `inputs_embeds`, so the entire pretrained transformer
stays usable while only a small input adapter and the CTC head need training.
`--freeze-backbone` then trains a small fraction of parameters (9.8% in a smoke
test: 38,752 frozen of 111,649).

**BERT has a fixed context length.** Long trials exceed `max_position_embeddings`,
so the convolutional front-end pools adaptively to fit, and the CTC length
mapping mirrors that same arithmetic. Verified exact at 800/1135/8000/64000
samples, all within the 512-frame limit.

**NeuroLM's NeuroProcessor is its vector-quantized EEG tokenizer.** In
`neurolm-adapter` the adapter is the analogue of NeuroLM's own
`encode_transform_layer`: a small learned map from processor features into the
language model's embedding space. `--processor scratch` trains that processor
from scratch instead of reusing the released weights.

Training prints the parameter budget so the effect of freezing is visible:

```
device=cpu mode=bert-ctc trials=559 channels=128 classes=1057
parameters: total=111,649 trainable=72,897 (65.3%) frozen=38,752
```

Checkpoints record the mode and backbone options, so `evaluate` rebuilds the
same architecture rather than guessing.

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
# 1. who is available, and how are the splits formed?
uv run bci-eeg-to-text split --dataset merged

# 2. extract (EEG, text) pairs. Raw data is huge; this is done once.
uv run bci-eeg-to-text prepare --dataset chineseeeg --out cache/trials
uv run bci-eeg-to-text prepare --dataset chisco     --out cache/trials
uv run bci-eeg-to-text prepare --dataset merged     --out cache/trials

# 3. what is in one run? reports markers vs text and whether they align
uv run bci-eeg-to-text inspect \
  eeg-text-bci-data/bci-eeg/ds005170/sub-01/ses-02/eeg/sub-01_ses-02_task-imagine_run-010_eeg.edf

# 4. train and evaluate (device auto-detects: cuda if usable, else cpu)
uv run bci-eeg-to-text train --cache cache/trials --out runs/exp1 --epochs 10
uv run bci-eeg-to-text evaluate --cache cache/trials --checkpoint runs/exp1/last.pt \
  --beam-width 8
```

## Tests

```bash
uv run pytest -q -m "not realdata"   # unit tests, no dataset needed
uv run pytest -q                     # includes dataset-backed tests
```

Real-data tests are marked `realdata` and skip automatically when the dataset is
not mounted. CI runs with `HF_HUB_OFFLINE=1`; the backbone tests build tiny
checkpoints locally rather than downloading them, and CI publishes coverage
(terminal, XML and HTML) plus JUnit results as artifacts on every run.

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
