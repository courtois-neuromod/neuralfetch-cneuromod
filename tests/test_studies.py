"""Smoke tests for all CNeuroMod study classes.

Verifies that:
- Every study class can be instantiated without errors.
- ``iter_timelines()`` returns the expected structure against the synthetic
  fixture (Friends study only, since it is the one populated by conftest).
- ``study_summary()`` returns a DataFrame with the expected columns.
- ``_load_timeline_events()`` returns a DataFrame with ``type``, ``start``,
  ``duration`` columns.
- The neuralset.studies entry-point discovers the package correctly.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import pandas as pd
import pytest

# Import all study classes
from neuralfetch_cneuromod.studies.anat import CNeuroModAnat
from neuralfetch_cneuromod.studies.floc import Floc
from neuralfetch_cneuromod.studies.friends import Friends
from neuralfetch_cneuromod.studies.gamepad import Gamepad
from neuralfetch_cneuromod.studies.harrypotter import HarryPotter
from neuralfetch_cneuromod.studies.hcptrt import HcpTrt
from neuralfetch_cneuromod.studies.mario import Mario
from neuralfetch_cneuromod.studies.movie10 import Movie10
from neuralfetch_cneuromod.studies.retinotopy import Retinotopy
from neuralfetch_cneuromod.studies.shinobi import Shinobi
from neuralfetch_cneuromod.studies.things import Things


# ---------------------------------------------------------------------------
# Entry-point discovery
# ---------------------------------------------------------------------------

class TestEntryPoint:
    """Verify neuralset.studies entry-point registration."""

    def test_entry_point_registered(self) -> None:
        """The package is registered under the neuralset.studies group."""
        eps = importlib.metadata.entry_points(group="neuralset.studies")
        values = [ep.value for ep in eps]
        assert "neuralfetch_cneuromod" in values, (
            f"neuralfetch_cneuromod not found in neuralset.studies entry-points.\n"
            f"Registered: {values}\n"
            "Did you install the package with `pip install -e .`?"
        )


# ---------------------------------------------------------------------------
# Instantiation smoke tests
# ---------------------------------------------------------------------------

#: All study classes that inherit from CNeuroModStudy (i.e. have BIDS + fMRIPrep).
FMRI_STUDY_CLASSES = [
    Friends, HarryPotter, Mario, Shinobi, Movie10, Things, Floc, HcpTrt,
    Retinotopy, Gamepad,
]


@pytest.mark.parametrize("StudyClass", FMRI_STUDY_CLASSES)
def test_instantiation(StudyClass: type, tmp_path: Path) -> None:
    """Every fMRI study class can be instantiated with a valid path.

    Parameters
    ----------
    StudyClass:
        Concrete study class to instantiate.
    tmp_path:
        Pytest-managed temporary directory used as the data root.
    """
    study = StudyClass(path=tmp_path)
    assert study.TASK, f"{StudyClass.__name__}.TASK must be a non-empty string"
    assert isinstance(study.bids_dir, Path)
    assert isinstance(study.fmriprep_dir, Path)


def test_anat_instantiation(tmp_path: Path) -> None:
    """CNeuroModAnat can be instantiated with a valid path."""
    study = CNeuroModAnat(path=tmp_path)
    assert isinstance(study.bids_dir, Path)
    assert isinstance(study.smriprep_dir, Path)


# ---------------------------------------------------------------------------
# iter_timelines — Friends synthetic fixture
# ---------------------------------------------------------------------------

class TestFriendsIterTimelines:
    """Tests for Friends.iter_timelines() using the synthetic fixture."""

    def test_yields_two_timelines(self, cneuromod_root: Path) -> None:
        """Friends yields exactly 2 timelines (2 runs × 1 subject × 1 session)."""
        study = Friends(path=cneuromod_root)
        timelines = list(study.iter_timelines())
        assert len(timelines) == 2

    def test_timeline_keys(self, cneuromod_root: Path) -> None:
        """Each timeline dict contains the expected keys."""
        study = Friends(path=cneuromod_root)
        for tl in study.iter_timelines():
            assert "subject" in tl
            assert "session" in tl
            assert "run" in tl
            assert "task" in tl
            assert tl["task"] == "friends"

    def test_subject_filter(self, cneuromod_root: Path) -> None:
        """subjects parameter restricts iteration to specified subjects."""
        study = Friends(path=cneuromod_root, subjects=["99"])
        assert list(study.iter_timelines()) == []

    def test_raises_if_no_fmriprep(self, tmp_path: Path) -> None:
        """iter_timelines() raises FileNotFoundError when fmriprep dir is absent."""
        study = Friends(path=tmp_path)
        with pytest.raises(FileNotFoundError, match="fMRIPrep directory not found"):
            list(study.iter_timelines())


# ---------------------------------------------------------------------------
# study_summary — Friends synthetic fixture
# ---------------------------------------------------------------------------

class TestFriendsStudySummary:
    """Tests for Friends.study_summary()."""

    def test_returns_dataframe(self, cneuromod_root: Path) -> None:
        """study_summary() returns a DataFrame with 2 rows."""
        study = Friends(path=cneuromod_root)
        summary = study.study_summary()
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2

    def test_subject_column(self, cneuromod_root: Path) -> None:
        """study_summary() prefixes subject with the class name."""
        study = Friends(path=cneuromod_root)
        summary = study.study_summary()
        assert all(summary["subject"].str.startswith("Friends/"))


# ---------------------------------------------------------------------------
# _load_timeline_events — Friends synthetic fixture
# ---------------------------------------------------------------------------

class TestFriendsLoadTimelineEvents:
    """Tests for Friends._load_timeline_events()."""

    def _first_timeline(self, cneuromod_root: Path) -> dict:
        """Return the first timeline from the Friends fixture."""
        study = Friends(path=cneuromod_root)
        return next(iter(study.iter_timelines()))

    def test_returns_dataframe(self, cneuromod_root: Path) -> None:
        """_load_timeline_events() returns a DataFrame."""
        study = Friends(path=cneuromod_root)
        tl = self._first_timeline(cneuromod_root)
        events = study._load_timeline_events(tl)
        assert isinstance(events, pd.DataFrame)

    def test_contains_fmri_row(self, cneuromod_root: Path) -> None:
        """_load_timeline_events() includes a row with type='Fmri'."""
        study = Friends(path=cneuromod_root)
        tl = self._first_timeline(cneuromod_root)
        events = study._load_timeline_events(tl)
        assert "Fmri" in events["type"].values

    def test_contains_stimulus_rows(self, cneuromod_root: Path) -> None:
        """_load_timeline_events() includes rows from the BIDS events TSV."""
        study = Friends(path=cneuromod_root)
        tl = self._first_timeline(cneuromod_root)
        events = study._load_timeline_events(tl)
        # 1 Fmri row + 5 stimulus rows (from conftest._write_events_tsv)
        assert len(events) == 6

    def test_required_columns(self, cneuromod_root: Path) -> None:
        """_load_timeline_events() always returns type, start, duration columns."""
        study = Friends(path=cneuromod_root)
        tl = self._first_timeline(cneuromod_root)
        events = study._load_timeline_events(tl)
        for col in ("type", "start", "duration"):
            assert col in events.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# CNeuroModAnat iter_timelines — synthetic fixture
# ---------------------------------------------------------------------------

class TestAnatIterTimelines:
    """Tests for CNeuroModAnat using a minimal synthetic fixture."""

    @pytest.fixture()
    def anat_root(self, tmp_path: Path) -> Path:
        """Create a minimal anatomy BIDS directory with one subject."""
        sub_dir = tmp_path / "CNeuroModAnat" / "bids" / "sub-01"
        sub_dir.mkdir(parents=True)
        return tmp_path

    def test_finds_one_subject(self, anat_root: Path) -> None:
        """iter_timelines() finds the single subject in the fixture."""
        study = CNeuroModAnat(path=anat_root)
        timelines = list(study.iter_timelines())
        assert len(timelines) == 1
        assert timelines[0]["subject"] == "01"

    def test_subject_filter(self, anat_root: Path) -> None:
        """subjects parameter filters correctly."""
        study = CNeuroModAnat(path=anat_root, subjects=["99"])
        assert list(study.iter_timelines()) == []

    def test_raises_if_no_dirs(self, tmp_path: Path) -> None:
        """iter_timelines() raises FileNotFoundError when no dirs exist."""
        study = CNeuroModAnat(path=tmp_path)
        with pytest.raises(FileNotFoundError):
            list(study.iter_timelines())


# ---------------------------------------------------------------------------
# Class variable sanity checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("StudyClass", FMRI_STUDY_CLASSES)
def test_class_variables_set(StudyClass: type, tmp_path: Path) -> None:
    """Every study class has non-empty TASK, BIDS_REPO, and FMRIPREP_REPO.

    Parameters
    ----------
    StudyClass:
        Concrete study class to check.
    tmp_path:
        Temporary directory (unused data path).
    """
    assert StudyClass.TASK, f"{StudyClass.__name__}.TASK is empty"
    assert StudyClass.BIDS_REPO, f"{StudyClass.__name__}.BIDS_REPO is empty"
    assert StudyClass.FMRIPREP_REPO, f"{StudyClass.__name__}.FMRIPREP_REPO is empty"
    assert StudyClass.description, f"{StudyClass.__name__}.description is empty"


@pytest.mark.parametrize("StudyClass", FMRI_STUDY_CLASSES)
def test_dataset_name_set(StudyClass: type, tmp_path: Path) -> None:
    """Every study class has a non-empty dataset_name class variable.

    Parameters
    ----------
    StudyClass:
        Concrete study class to check.
    tmp_path:
        Temporary directory (unused data path).
    """
    assert StudyClass.dataset_name, f"{StudyClass.__name__}.dataset_name is empty"
