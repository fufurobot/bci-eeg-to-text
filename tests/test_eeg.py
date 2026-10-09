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

#: Marker channel name used by the dataset.
TRIGGER = "Trigger"


def write_tiny_edf(path, *, include_trigger: bool = True):
    """Write a minimal valid EDF file with pyedflib.

    Using the real library keeps these tests honest: the file must be readable
    by the same code path that reads the dataset.
    """
    import pyedflib

    n_samples = 200
    labels = ["EEG1", "EEG2"] + ([TRIGGER] if include_trigger else [])
    signal_headers = [
        pyedflib.highlevel.make_signal_header(
            label,
            sample_frequency=100,
            physical_min=-200.0,
            physical_max=200.0,
            digital_min=-32768,
            digital_max=32767,
        )
        for label in labels
    ]
    data = [np.zeros(n_samples, dtype=float) for _ in labels]
    with pyedflib.EdfWriter(str(path), len(labels), file_type=pyedflib.FILETYPE_EDFPLUS) as writer:
        writer.setSignalHeaders(signal_headers)
        writer.writeSamples(data)
    return path


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

    def test_contiguous_run_is_one_pulse(self) -> None:
        """Indices 5..14 are one unbroken run of the same value.

        They are physically indistinguishable from a single 10-sample pulse, so
        they must count once. (Real pulses are ~100 samples wide and separated
        by ~10 s, so they never touch.)
        """
        trig = np.zeros(30, dtype=float)
        trig[5:15] = REWARD_PULSE
        assert find_pulses(trig).tolist() == [5]

    def test_single_sample_gap_yields_two_pulses(self) -> None:
        """A one-sample non-pulse gap is a real boundary."""
        trig = np.zeros(30, dtype=float)
        trig[5:10] = REWARD_PULSE
        trig[11:15] = REWARD_PULSE
        assert find_pulses(trig).tolist() == [5, 11]

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


class TestAnnexPointers:
    """The Chisco release ships many runs as unfetched git-annex pointers."""

    def test_detects_pointer_file(self, tmp_path) -> None:
        from bci_eeg_to_text.eeg import is_annex_pointer

        ptr = tmp_path / "sub-01_ses-01_task-imagine_run-01_eeg.edf"
        ptr.write_text(
            "/annex/objects/SHA256E-s644413760--1a7392a6fdd8aa2b6b518c92198224f8e5ed9bb.edf\n"
        )
        assert is_annex_pointer(ptr)

    def test_real_edf_is_not_a_pointer(self, tmp_path) -> None:
        from bci_eeg_to_text.eeg import is_annex_pointer

        real = tmp_path / "real.edf"
        real.write_bytes(b"0       " + b"\x00" * 5000)
        assert not is_annex_pointer(real)

    def test_missing_file_is_not_a_pointer(self, tmp_path) -> None:
        from bci_eeg_to_text.eeg import is_annex_pointer

        assert not is_annex_pointer(tmp_path / "nope.edf")

    def test_require_real_data_raises_helpfully(self, tmp_path) -> None:
        from bci_eeg_to_text.eeg import MissingDataError, require_real_data

        ptr = tmp_path / "stub.edf"
        ptr.write_text("/annex/objects/SHA256E-s123--abc.edf\n")
        with pytest.raises(MissingDataError, match="unfetched git-annex pointer"):
            require_real_data(ptr)

    def test_require_real_data_passes_for_readable_edf(self, tmp_path) -> None:
        """A file pyedflib can genuinely open must not be flagged as missing."""
        from bci_eeg_to_text.eeg import require_real_data

        require_real_data(write_tiny_edf(tmp_path / "tiny.edf"))

    def test_require_real_data_rejects_unreadable_file(self, tmp_path) -> None:
        """A truncated download must be reported, not parsed as data."""
        from bci_eeg_to_text.eeg import MissingDataError, require_real_data

        junk = tmp_path / "truncated.edf"
        junk.write_bytes(b"0       " + b"\x00" * 5000)
        with pytest.raises(MissingDataError, match="cannot be read as EDF"):
            require_real_data(junk)

    def test_is_readable_edf_is_false_for_pointer(self, tmp_path) -> None:
        from bci_eeg_to_text.eeg import is_readable_edf

        ptr = tmp_path / "stub.edf"
        ptr.write_text("/annex/objects/SHA256E-s123--abc.edf\n")
        assert not is_readable_edf(ptr)

    def test_is_readable_edf_requires_the_trigger_channel(self, tmp_path) -> None:
        """A recording without a Trigger channel cannot be aligned."""
        from bci_eeg_to_text.eeg import is_readable_edf

        edf = write_tiny_edf(tmp_path / "no_trigger.edf", include_trigger=False)
        assert not is_readable_edf(edf)

    def test_is_readable_edf_true_for_valid_file(self, tmp_path) -> None:
        from bci_eeg_to_text.eeg import is_readable_edf

        assert is_readable_edf(write_tiny_edf(tmp_path / "ok.edf"))


@pytest.mark.realdata
class TestDatasetInventory:
    def test_reports_pointer_and_real_run_counts(self) -> None:
        """Documents the 46-stub / 177-real split of this release."""
        from pathlib import Path

        from bci_eeg_to_text.eeg import is_annex_pointer, iter_runs, usable_runs

        root = Path(DATASET)
        if not root.exists():
            pytest.skip("dataset not mounted")
        all_runs = list(iter_runs(root))
        usable = usable_runs(root)
        assert len(all_runs) == 223, f"expected 223 runs, found {len(all_runs)}"
        assert all(not is_annex_pointer(p) for p in usable)
        assert len(usable) == 177, f"expected 177 usable runs, found {len(usable)}"
