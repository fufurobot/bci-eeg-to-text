"""Tests for EDF reading and reward-pulse alignment."""

from __future__ import annotations

import numpy as np
import pytest

from bci_eeg_to_text.eeg import (
    REWARD_PULSE,
    AlignmentError,
    RunId,
    align_segments,
    find_pulses,
    segment_bounds,
)

DATASET = "eeg-text-bci-data/bci-eeg/ds005170"


class TestRunId:
    def test_parses_bids_filename(self) -> None:
        rid = RunId.parse("sub-01_ses-02_task-imagine_run-010_eeg.edf")
        assert rid == RunId(subject="sub-01", session=2, task="imagine", run=10)

    def test_run_index_is_global_not_per_session(self) -> None:
        """run-08 and run-010 both occur in ses-02; the number is global."""
        a = RunId.parse("sub-01_ses-02_task-imagine_run-08_eeg.edf")
        b = RunId.parse("sub-01_ses-02_task-imagine_run-010_eeg.edf")
        assert (a.session, a.run) == (2, 8)
        assert (b.session, b.run) == (2, 10)

    def test_key_is_zero_padded_and_stable(self) -> None:
        rid = RunId.parse("sub-01_ses-02_task-imagine_run-8_eeg.edf")
        assert rid.key == "sub-01_ses-02_task-imagine_run-008"

    def test_rejects_non_bids_name(self) -> None:
        with pytest.raises(ValueError, match="not a BIDS EEG filename"):
            RunId.parse("random.edf")

    def test_accepts_full_path(self) -> None:
        rid = RunId.parse("/data/sub-03/ses-01/eeg/sub-03_ses-01_task-imagine_run-001_eeg.edf")
        assert rid.subject == "sub-03"


class TestFindPulses:
    def test_finds_single_pulse_onset(self) -> None:
        trig = np.zeros(100, dtype=float)
        trig[10:20] = REWARD_PULSE
        assert find_pulses(trig).tolist() == [10]

    def test_onset_not_middle_of_pulse(self) -> None:
        """Segmenting from the middle of a pulse would shift the trial."""
        trig = np.zeros(50, dtype=float)
        trig[30:40] = REWARD_PULSE
        assert find_pulses(trig)[0] == 30

    def test_finds_multiple_pulses_in_order(self) -> None:
        trig = np.zeros(1000, dtype=float)
        for start in (100, 400, 900):
            trig[start : start + 10] = REWARD_PULSE
        assert find_pulses(trig).tolist() == [100, 400, 900]

    def test_adjacent_pulses_are_separate(self) -> None:
        """A pulse immediately after another is a distinct trial."""
        trig = np.zeros(30, dtype=float)
        trig[5:10] = REWARD_PULSE
        trig[10:15] = REWARD_PULSE
        assert find_pulses(trig).tolist() == [5, 10]

    def test_ignores_other_marker_values(self) -> None:
        """65280 is the baseline marker and must not be treated as a pulse."""
        trig = np.full(100, 65280.0)
        trig[50:60] = REWARD_PULSE
        assert find_pulses(trig).tolist() == [50]

    def test_no_pulses_returns_empty(self) -> None:
        assert find_pulses(np.zeros(100)).size == 0

    def test_pulse_at_sample_zero_is_detected(self) -> None:
        trig = np.zeros(20, dtype=float)
        trig[0:5] = REWARD_PULSE
        assert find_pulses(trig).tolist() == [0]

    def test_rejects_2d_input(self) -> None:
        with pytest.raises(ValueError, match="must be 1-D"):
            find_pulses(np.zeros((2, 10)))


class TestSegmentBounds:
    def test_next_onset_defines_the_end(self) -> None:
        bounds = segment_bounds([0, 100], n_samples=200)
        assert bounds == [(0, 100), (100, 200)]

    def test_pre_and_post_samples_are_applied(self) -> None:
        bounds = segment_bounds([100], n_samples=500, pre_samples=10, post_samples=50)
        assert bounds == [(90, 150)]

    def test_start_is_clipped_to_zero(self) -> None:
        bounds = segment_bounds([5], n_samples=100, pre_samples=50, post_samples=10)
        assert bounds == [(0, 15)]

    def test_final_segment_is_clipped_to_n_samples(self) -> None:
        bounds = segment_bounds([90], n_samples=100, post_samples=50)
        assert bounds == [(90, 100)]

    def test_rejects_non_increasing_onsets(self) -> None:
        with pytest.raises(AlignmentError, match="strictly increasing"):
            segment_bounds([100, 100], n_samples=200)


class TestAlignSegments:
    def test_pairs_onsets_with_texts(self) -> None:
        segs = align_segments([0, 100, 200], ["a", "b", "c"], n_samples=300)
        assert [s.text for s in segs] == ["a", "b", "c"]
        assert [s.index for s in segs] == [0, 1, 2]

    def test_mismatch_raises_instead_of_truncating(self) -> None:
        """Silently pairing a subset would train on mislabelled data."""
        with pytest.raises(AlignmentError, match="refusing to train on misaligned pairs"):
            align_segments([0, 100], ["a", "b", "c"], n_samples=300)

    def test_mismatch_raises_when_fewer_texts_than_pulses(self) -> None:
        with pytest.raises(AlignmentError, match="2.*1|1.*2"):
            align_segments([0, 100], ["a"], n_samples=300)

    def test_error_message_reports_both_counts(self) -> None:
        with pytest.raises(AlignmentError) as exc:
            align_segments([0], ["a", "b"], n_samples=100)
        assert "1" in str(exc.value) and "2" in str(exc.value)

    def test_segments_are_contiguous_and_ordered(self) -> None:
        segs = align_segments([0, 100, 250], ["a", "b", "c"], n_samples=400)
        for a, b in zip(segs, segs[1:], strict=False):
            assert a.stop_sample == b.start_sample

    def test_rejects_empty_segment(self) -> None:
        with pytest.raises(AlignmentError, match="empty"):
            align_segments([10], ["a"], n_samples=100, post_samples=0)

    def test_pulse_count_matches_text_count_is_the_contract(self) -> None:
        segs = align_segments(range(0, 300, 10), ["x"] * 30, n_samples=300)
        assert len(segs) == 30


@pytest.mark.realdata
class TestAgainstRealDataset:
    """Checks the contract against the read-only dataset mount."""

    def test_run_010_has_150_pulses_matching_split_data_2(self) -> None:
        """The end-to-end alignment claim from issue #1."""
        from pathlib import Path

        from bci_eeg_to_text.textdata import read_split

        edf = Path(DATASET) / "sub-01/ses-02/eeg/sub-01_ses-02_task-imagine_run-010_eeg.edf"
        if not edf.exists():
            pytest.skip("dataset not mounted")

        from bci_eeg_to_text.eeg import read_signals

        trig, rate = read_signals(edf, ["Trigger"])
        assert rate == 1000.0
        pulses = find_pulses(trig[0])
        texts = read_split(Path(DATASET) / "textdataset/split_data_2.xlsx")
        assert len(pulses) == len(texts) == 150
        # And alignment must succeed rather than raise.
        segs = align_segments(pulses, texts, n_samples=trig.shape[1])
        assert len(segs) == 150
