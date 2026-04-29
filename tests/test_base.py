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
    bold_path,
    confounds_path,
    events_path,
    get_bold_runs,
    get_sessions,
    get_subjects,
    iter_bids_runs,
    load_bold_masked,
    load_confounds_tsv,
    load_events_tsv,
    mask_path,
)


# ---------------------------------------------------------------------------
# bold_path
# ---------------------------------------------------------------------------

class TestBoldPath:
    """Tests for the bold_path() path builder."""

    def test_with_session_and_run(self, tmp_path: Path) -> None:
        """bold_path() includes ses- and run- entities when provided."""
        result = bold_path(tmp_path, "01", "friends", session="001", run=1)
        assert "sub-01" in result.parts
        assert result.name == (
            "sub-01_ses-001_task-friends_run-1"
            "_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
        )

    def test_without_session(self, tmp_path: Path) -> None:
        """bold_path() omits ses- entity when session=None."""
        result = bold_path(tmp_path, "01", "friends", session=None, run=1)
        assert "ses" not in result.name

    def test_without_run(self, tmp_path: Path) -> None:
        """bold_path() omits run- entity when run=None."""
        result = bold_path(tmp_path, "01", "friends", session="001", run=None)
        assert "run" not in result.name

    def test_custom_space(self, tmp_path: Path) -> None:
        """bold_path() uses custom space and resolution."""
        result = bold_path(
            tmp_path, "01", "friends",
            session="001", run=1,
            space="T1w", resolution="native",
        )
        assert "space-T1w" in result.name
        assert "res-native" in result.name


# ---------------------------------------------------------------------------
# confounds_path
# ---------------------------------------------------------------------------

class TestConfoundsPath:
    """Tests for the confounds_path() path builder."""

    def test_name_format(self, tmp_path: Path) -> None:
        """confounds_path() produces correct filename."""
        result = confounds_path(tmp_path, "01", "friends", session="001", run=2)
        assert result.name == (
            "sub-01_ses-001_task-friends_run-2_desc-confounds_timeseries.tsv"
        )

    def test_no_run(self, tmp_path: Path) -> None:
        """confounds_path() omits run entity when run=None."""
        result = confounds_path(tmp_path, "01", "friends", session="001", run=None)
        assert "run" not in result.name


# ---------------------------------------------------------------------------
# mask_path
# ---------------------------------------------------------------------------

class TestMaskPath:
    """Tests for the mask_path() path builder."""

    def test_name_format(self, tmp_path: Path) -> None:
        """mask_path() produces correct filename."""
        result = mask_path(tmp_path, "01", "friends", session="001", run=1)
        assert "desc-brain_mask" in result.name
        assert result.suffix == ".gz"


# ---------------------------------------------------------------------------
# events_path
# ---------------------------------------------------------------------------

class TestEventsPath:
    """Tests for the events_path() path builder."""

    def test_name_format(self, tmp_path: Path) -> None:
        """events_path() produces correct BIDS events TSV filename."""
        result = events_path(tmp_path, "01", "friends", session="001", run=1)
        assert result.name == "sub-01_ses-001_task-friends_run-1_events.tsv"


# ---------------------------------------------------------------------------
# get_subjects / get_sessions
# ---------------------------------------------------------------------------

class TestEntityDiscovery:
    """Tests for subject/session discovery helpers."""

    def test_get_subjects(self, cneuromod_root: Path) -> None:
        """get_subjects() finds sub-01 in the synthetic fixture."""
        fmriprep_dir = cneuromod_root / "Friends" / "fmriprep"
        subjects = get_subjects(fmriprep_dir)
        assert subjects == ["01"]

    def test_get_subjects_empty(self, tmp_path: Path) -> None:
        """get_subjects() returns empty list for an empty directory."""
        assert get_subjects(tmp_path) == []

    def test_get_sessions(self, cneuromod_root: Path) -> None:
        """get_sessions() finds ses-001 in the synthetic fixture."""
        fmriprep_dir = cneuromod_root / "Friends" / "fmriprep"
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
        fmriprep_dir = cneuromod_root / "Friends" / "fmriprep"
        runs = get_bold_runs(fmriprep_dir, "01", "friends", session="001")
        assert sorted(runs) == ["1", "2"]

    def test_no_runs_for_missing_session(self, cneuromod_root: Path) -> None:
        """get_bold_runs() returns [] for a non-existent session."""
        fmriprep_dir = cneuromod_root / "Friends" / "fmriprep"
        runs = get_bold_runs(fmriprep_dir, "01", "friends", session="999")
        assert runs == []


# ---------------------------------------------------------------------------
# iter_bids_runs
# ---------------------------------------------------------------------------

class TestIterBidsRuns:
    """Tests for iter_bids_runs()."""

    def test_yields_correct_triples(self, cneuromod_root: Path) -> None:
        """iter_bids_runs() yields (sub, ses, run, task) for all BOLD files."""
        fmriprep_dir = cneuromod_root / "Friends" / "fmriprep"
        triples = list(iter_bids_runs(fmriprep_dir, None, "friends"))
        assert len(triples) == 2
        for t in triples:
            assert t["subject"] == "01"
            assert t["session"] == "001"
            assert t["task"] == "friends"
            assert t["run"] in ("1", "2")

    def test_subject_filter(self, cneuromod_root: Path) -> None:
        """iter_bids_runs() respects the subjects filter."""
        fmriprep_dir = cneuromod_root / "Friends" / "fmriprep"
        triples = list(
            iter_bids_runs(fmriprep_dir, None, "friends", subjects=["99"])
        )
        assert triples == []


# ---------------------------------------------------------------------------
# load_events_tsv
# ---------------------------------------------------------------------------

class TestLoadEventsTsv:
    """Tests for load_events_tsv()."""

    def test_loads_fixture(self, cneuromod_root: Path) -> None:
        """load_events_tsv() reads the synthetic events file correctly."""
        ep = events_path(
            cneuromod_root / "Friends" / "bids",
            "01", "friends", session="001", run=1,
        )
        df = load_events_tsv(ep)
        assert "onset" in df.columns
        assert "duration" in df.columns
        assert len(df) == 5

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        """load_events_tsv() raises FileNotFoundError for non-existent path."""
        with pytest.raises(FileNotFoundError):
            load_events_tsv(tmp_path / "nonexistent.tsv")

    def test_raises_on_missing_columns(self, tmp_path: Path) -> None:
        """load_events_tsv() raises ValueError when required columns are absent."""
        bad = tmp_path / "bad_events.tsv"
        bad.write_text("col_a\tcol_b\n1\t2\n")
        with pytest.raises(ValueError, match="missing required columns"):
            load_events_tsv(bad)


# ---------------------------------------------------------------------------
# load_confounds_tsv
# ---------------------------------------------------------------------------

class TestLoadConfoundsTsv:
    """Tests for load_confounds_tsv()."""

    def test_loads_all_columns(self, cneuromod_root: Path) -> None:
        """load_confounds_tsv() returns all columns when no filter is applied."""
        cp = confounds_path(
            cneuromod_root / "Friends" / "fmriprep",
            "01", "friends", session="001", run=1,
        )
        df = load_confounds_tsv(cp)
        assert len(df) == 20  # 20 volumes
        assert "trans_x" in df.columns

    def test_column_filter(self, cneuromod_root: Path) -> None:
        """load_confounds_tsv() returns only requested columns."""
        cp = confounds_path(
            cneuromod_root / "Friends" / "fmriprep",
            "01", "friends", session="001", run=1,
        )
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
        bp = bold_path(
            cneuromod_root / "Friends" / "fmriprep",
            "01", "friends", session="001", run=1,
        )
        arr = load_bold_masked(bp, mask_path_=None)
        assert arr.ndim == 2
        assert arr.shape[1] == 20  # n_volumes

    def test_loads_with_mask(self, cneuromod_root: Path) -> None:
        """load_bold_masked() applies brain mask and reduces spatial dimension."""
        fmriprep_dir = cneuromod_root / "Friends" / "fmriprep"
        bp = bold_path(fmriprep_dir, "01", "friends", session="001", run=1)
        mp = mask_path(fmriprep_dir, "01", "friends", session="001", run=1)
        arr_no_mask = load_bold_masked(bp, mask_path_=None)
        arr_masked = load_bold_masked(bp, mask_path_=mp)
        # Masked array should have fewer voxels than the full flat array
        assert arr_masked.shape[0] < arr_no_mask.shape[0]
        assert arr_masked.shape[1] == 20

    def test_raises_on_missing_bold(self, tmp_path: Path) -> None:
        """load_bold_masked() raises FileNotFoundError when BOLD is absent."""
        with pytest.raises(FileNotFoundError, match="BOLD file not found"):
            load_bold_masked(tmp_path / "nonexistent.nii.gz")
