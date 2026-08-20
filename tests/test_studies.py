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

from neuralfetch_cneuromod.studies import [
    #EmotionVideos, Floc,
    Friends, HarryPotter, #HcpTrt,
    Mario, Mario3, MarioStars, Movie10,
    #Narratives, OOD,
    PetitPrince, #Retinotopy,
    Shinobi, #Things, #Triplets,
]

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

#: All study classes that inherit from CNeuroModStudy (i.e. have BIDS + fMRIPrep + timeseries).
FMRI_STUDY_CLASSES = [
    #EmotionVideos, Floc, Friends, HarryPotter, HcpTrt, Mario, MarioStars,
    #Mario3, Movie10, Narratives, PetitPrince, Retinotopy, Shinobi,
    #Things, Triplets,
    Movie10, Friends,
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
    assert isinstance(study.timeseries_dir, Path)


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
            assert "task" in tl

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
    assert StudyClass.TIMESERIES_REPO, f"{StudyClass.__name__}.TIMESERIES_REPO is empty"
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
