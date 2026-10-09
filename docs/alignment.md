# Dataset alignment verification

Reproducible evidence for the alignment rule the whole project rests on:
**run `run-NNN` pairs with `textdataset/split_data_N.xlsx`, and the i-th reward
pulse corresponds to the i-th sentence in that file.**

Tracking issue: [#1](https://github.com/fufurobot/bci-eeg-to-text/issues/1)
Data-quality issue: [#2](https://github.com/fufurobot/bci-eeg-to-text/issues/2)

## Why this needs verifying

`ds005170` is **output-only** EEG. Its marker channel carries a single uniform
reward pulse per trial and **no stimulus-onset triggers**, so the usual BIDS/MNE
event route cannot say which sentence was imagined when.

If the recovered pairing is wrong, the model trains on mislabelled data and
reports near-chance results that look like a modelling failure rather than an
alignment bug. That is why the pipeline raises on a count mismatch instead of
truncating or padding to fit.

## Method

```bash
uv run python scripts/verify_alignment.py            # sampled
uv run python scripts/verify_alignment.py --all      # every run
```

The script reads only the `Trigger` channel and compares its pulse count with
the row count of the matching `split_data_N.xlsx`. Reading one channel is far
cheaper than the full 128-channel read, but on a slow mount even this is minutes
per run (measured ~1 MB/s, so a 574 MB file takes ~1-4 minutes).

## Results

### Confirmed alignment

| run | reward pulses | stimulus sentences | aligned |
| --- | --- | --- | --- |
| `sub-01_ses-02_task-imagine_run-010` | 150 | 150 | yes |
| `sub-01_ses-01_task-imagine_run-01` | 200 | 200 | yes |

`inspect` reports this directly:

```console
$ uv run bci-eeg-to-text inspect .../sub-01_ses-02_task-imagine_run-010_eeg.edf
{
  "run": "sub-01_ses-02_task-imagine_run-010",
  "sample_rate_hz": 1000.0,
  "duration_s": 1736.4,
  "n_eeg_channels": 128,
  "reward_pulses": 150,
  "stimulus_sentences": 150,
  "aligned": true,
  "stimulus_file": "split_data_10.xlsx",
  "mean_trial_gap_s": 10.08
}
```

### Dataset inventory

Not every path is usable data. Classification of all 223 EDF paths:

| category | count | how it is detected |
| --- | --- | --- |
| unfetched git-annex pointer (104 B) | 46 | `eeg.is_annex_pointer` |
| truncated partial download | 9 | `eeg.is_readable_edf` |
| usable recording | **168** | — |

`sub-03` contributes 45 of the 46 pointers, so that subject is entirely absent
locally. The 9 truncated files are ~38 MB each but end early; they carry a valid
EDF header, so only actually opening them reveals the problem.

## Structural facts that constrain the code

- Runs carry a **global** run index: `run-08` and `run-010` both live in
  `ses-02`, so mapping keys on the run number, never on session order.
- All 223 runs are `task-imagine`. Despite the derivatives listing a
  `task-read`, **no `task-read` EDF exists**.
- Raw EDF: 128 EEG channels plus `VEO`/`HEO`/`EKG`/`EMG`/`Trigger`, 1000 Hz.
- Trials are ~10 s apart; `split_data_1.xlsx` has 200 sentences while files
  2..44 have 150 and file 45 has 31.

## Assumption still open

The reward pulse is treated as marking trial **onset**. If it marks the trial
*end* instead, segment windows are shifted. The offset is configurable via
`ExtractConfig.pre_samples` / `post_samples`, and the default assumption is
stated here so it can be challenged.
