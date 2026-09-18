"""Unit tests for neuralfetch_cneuromod._utils helpers.

Tests are strictly offline (no DataLad, no network): they exercise path
builders, entity discovery, and TSV loaders against the synthetic fixture
directory created by ``conftest.py``.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from neuralfetch_cneuromod._utils import (
    get_bold_runs,#
    get_sessions,#
    get_subjects,#
    iter_bids_runs,
    iter_tseries_runs,
    load_bold_masked,
    load_confounds_tsv,
    datalad_get_list,
)

# ---------------------------------------------------------------------------
# get_subjects / get_sessions
# ---------------------------------------------------------------------------

class TestEntityDiscovery:
    """Tests for subject/session discovery helpers."""

    def test_get_subjects(self, cneuromod_root: Path) -> None:
        """get_subjects() finds sub-01 in the synthetic fixture."""
        fmriprep_dir = cneuromod_root / "friends" / "fmriprep"
        subjects = get_subjects(fmriprep_dir)
        assert subjects == ["01"]

    def test_get_subjects_empty(self, tmp_path: Path) -> None:
        """get_subjects() returns empty list for an empty directory."""
        assert get_subjects(tmp_path) == []

    def test_get_sessions(self, cneuromod_root: Path) -> None:
        """get_sessions() finds ses-001 in the synthetic fixture."""
        fmriprep_dir = cneuromod_root / "friends" / "fmriprep"
        sessions = get_sessions(fmriprep_dir, "01")
        assert sessions == ["001"]

    def test_get_sessions_missing_subject(self, tmp_path: Path) -> None:
        """get_sessions() returns [] for a non-existent subject."""
        assert get_sessions(tmp_path, "99") == []


# ---------------------------------------------------------------------------
# get_bold_runs
# ---------------------------------------------------------------------------

class TestGetBoldRuns:
    """Tests for get_bold_runs()."""

    def test_finds_two_runs(self, cneuromod_root: Path) -> None:
        """get_bold_runs() detects two runs in the synthetic fixture."""
        fmriprep_dir = cneuromod_root / "friends" / "fmriprep"
        tasks = [t for t, _, _ in get_bold_runs(
            fmriprep_dir, "01", session="001")]
        assert sorted(tasks) == ["s01e01a", "s01e01b"]

    def test_no_runs_for_missing_session(self, cneuromod_root: Path) -> None:
        """get_bold_runs() returns [] for a non-existent session."""
        fmriprep_dir = cneuromod_root / "friends" / "fmriprep"
        tasks = [t for t, _, _ in get_bold_runs(
            fmriprep_dir, "01", session="999")]
        assert tasks == []


# ---------------------------------------------------------------------------
# iter_bids_runs
# ---------------------------------------------------------------------------

class TestIterBidsRuns:
    """Tests for iter_bids_runs()."""

    def test_yields_correct_triples(self, cneuromod_root: Path) -> None:
        """iter_bids_runs() yields (sub, bold_path, ses, task, run) for all BOLD files."""
        fmriprep_dir = cneuromod_root / "friends" / "fmriprep"
        bids_runs = list(iter_bids_runs(fmriprep_dir, subjects=None))
        assert len(bids_runs) == 2
        for br in bids_runs:
            assert br["subject"] == "01"
            assert br["session"] == "001"
            assert "s01e01" in br["task"]
            assert br["run"] is None

    def test_subject_filter(self, cneuromod_root: Path) -> None:
        """iter_bids_runs() respects the subjects filter."""
        fmriprep_dir = cneuromod_root / "friends" / "fmriprep"
        bids_runs = list(
            iter_bids_runs(fmriprep_dir, subjects=["99"])
        )
        assert bids_runs == []


# ---------------------------------------------------------------------------
# iter_tseries_runs
# ---------------------------------------------------------------------------

class TestIterTseriesRuns:
    """Tests for iter_tseries_runs()."""

    def test_yields_h5_runs(self, cneuromod_root: Path) -> None:
        """iter_tseries_runs() yields (sub, hdf5_path, ses, task, run) for all
        BOLD timeseries."""
        timeseries_dir = cneuromod_root / "friends" / "timeseries" 
        ts_runs = list(iter_tseries_runs(timeseries_dir, task="friends", subjects=None))
        assert len(ts_runs) == 2
        for ts in ts_runs:
            assert ts["subject"] == "01"
            assert ts["session"] == "ses-001"
            assert ts["task"] is None
            assert "s01e01" in ts["run"]


# ---------------------------------------------------------------------------
# load_confounds_tsv
# ---------------------------------------------------------------------------

class TestLoadConfoundsTsv:
    """Tests for load_confounds_tsv()."""

    def test_loads_all_columns(self, cneuromod_root: Path) -> None:
        """load_confounds_tsv() returns all columns when no filter is applied."""
        fmriprep_dir = cneuromod_root / "friends" / "fmriprep"
        bids_runs = list(iter_bids_runs(fmriprep_dir, subjects=None))
        for br in bids_runs:
            cp = Path(br['file_path'].split("_space")[0].replace(
                "_part-mag", "") + "_desc-confounds_timeseries.tsv")
            df = load_confounds_tsv(cp)
            assert len(df) == 20  # 20 volumes
            assert "trans_x" in df.columns

    def test_column_filter(self, cneuromod_root: Path) -> None:
        """load_confounds_tsv() returns only requested columns."""
        fmriprep_dir = cneuromod_root / "friends" / "fmriprep"
        bids_runs = list(iter_bids_runs(fmriprep_dir, subjects=None))
        for br in bids_runs:
            cp = Path(br['file_path'].split("_space")[0].replace(
                "_part-mag", "") + "_desc-confounds_timeseries.tsv")
            df = load_confounds_tsv(cp, columns=["trans_x", "trans_y"])
            assert list(df.columns) == ["trans_x", "trans_y"]

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        """load_confounds_tsv() raises FileNotFoundError for non-existent path."""
        with pytest.raises(FileNotFoundError):
            load_confounds_tsv(tmp_path / "missing.tsv")


# ---------------------------------------------------------------------------
# load_bold_masked
# ---------------------------------------------------------------------------

class TestLoadBoldMasked:
    """Tests for load_bold_masked()."""

    def test_loads_unmasked(self, cneuromod_root: Path) -> None:
        """load_bold_masked() returns (n_voxels, n_volumes) without a mask."""
        fmriprep_dir = cneuromod_root / "friends" / "fmriprep"
        bids_runs = list(iter_bids_runs(fmriprep_dir, subjects=None))
        for br in bids_runs:
            arr = load_bold_masked(Path(br['file_path']), mask_path_=None)
            assert arr.ndim == 2
            assert arr.shape[1] == 20  # n_volumes

    def test_loads_with_mask(self, cneuromod_root: Path) -> None:
        """load_bold_masked() applies brain mask and reduces spatial dimension."""
        fmriprep_dir = cneuromod_root / "friends" / "fmriprep"
        bids_run = list(iter_bids_runs(fmriprep_dir, subjects=None))[0]
        bp = Path(bids_run["file_path"])
        mp = Path(bids_run["file_path"].replace("desc-preproc_bold", "desc-brain_mask"))
        arr_no_mask = load_bold_masked(bp, mask_path_=None)
        arr_masked = load_bold_masked(bp, mask_path_= mp)
        # Masked array should have fewer voxels than the full flat array
        assert arr_masked.shape[0] < arr_no_mask.shape[0]
        assert arr_masked.shape[1] == 20

    def test_raises_on_missing_bold(self, tmp_path: Path) -> None:
        """load_bold_masked() raises FileNotFoundError when BOLD is absent."""
        with pytest.raises(FileNotFoundError, match="BOLD file not found"):
            load_bold_masked(tmp_path / "nonexistent.nii.gz")
