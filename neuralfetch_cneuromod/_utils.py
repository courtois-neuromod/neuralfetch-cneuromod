"""Utility helpers shared across CNeuroMod study classes.

Provides file-path builders following the fMRIPrep BIDS-derivatives naming
convention and lightweight BIDS entity parsers.  All path helpers are pure
functions that take BIDS entities as keyword arguments and return
:class:`~pathlib.Path` objects — no IO is performed until the caller
explicitly opens the returned path.

DataLad operations (clone + get) are handled by
:class:`neuralfetch.download.Datalad`; this module does *not* invoke DataLad
or git-annex directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BIDS / fMRIPrep path builders
# ---------------------------------------------------------------------------

#: Default MNI152 template identifier used by fMRIPrep.
DEFAULT_SPACE = "MNI152NLin2009cAsym"
#: Default resolution string used by fMRIPrep.
DEFAULT_RESOLUTION = None


def bold_path(
    fmriprep_dir: Path,
    subject: str,
    task: str,
    *,
    session: str | None = None,
    run: str | int | None = None,
    space: str = DEFAULT_SPACE,
    resolution: str = DEFAULT_RESOLUTION,
    suffix: str = "bold",
    extension: str = ".nii.gz",
) -> Path:
    """Return the expected path to a fMRIPrep preprocessed BOLD file.

    Parameters
    ----------
    fmriprep_dir:
        Root of the fMRIPrep derivatives dataset (i.e. the directory that
        contains ``sub-*`` folders).
    subject:
        Subject label *without* the ``sub-`` prefix (e.g. ``"01"``).
    task:
        BIDS task label (e.g. ``"friends"``).
    session:
        BIDS session label without the ``ses-`` prefix.  When ``None`` the
        path is built without a session level.
    run:
        Run index (1-based integer) or string label.  When ``None`` the
        ``run-`` entity is omitted from the filename.
    space:
        fMRIPrep output space template (default ``"MNI152NLin2009cAsym"``).
    resolution:
        Template resolution label (default ``"2"``).
    suffix:
        BIDS suffix (default ``"bold"``).
    extension:
        File extension (default ``".nii.gz"``).

    Returns
    -------
    Path
        Full path to the expected BOLD NIfTI file.
    """
    sub_dir = fmriprep_dir / f"sub-{subject}"
    if session is not None:
        sub_dir = sub_dir / f"ses-{session}" / "func"
    else:
        sub_dir = sub_dir / "func"

    entities: list[str] = [f"sub-{subject}"]
    if session is not None:
        entities.append(f"ses-{session}")
    entities.append(f"task-{task}")
    if run is not None:
        entities.append(f"run-{run}")
    if space is not None:
        entities.append(f"space-{space}")
    if resolution is not None:
        entities.append(f"res-{resolution}")
    entities.append(f"desc-preproc_{suffix}")

    fname = "_".join(entities) + extension
    return sub_dir / fname


def confounds_path(
    fmriprep_dir: Path,
    subject: str,
    task: str,
    *,
    session: str | None = None,
    run: str | int | None = None,
) -> Path:
    """Return the expected path to a fMRIPrep confounds timeseries TSV.

    Parameters
    ----------
    fmriprep_dir:
        Root of the fMRIPrep derivatives dataset.
    subject:
        Subject label without the ``sub-`` prefix.
    task:
        BIDS task label.
    session:
        BIDS session label without ``ses-`` prefix.
    run:
        Run index or label; omitted when ``None``.

    Returns
    -------
    Path
        Full path to the expected confounds TSV file.
    """
    sub_dir = fmriprep_dir / f"sub-{subject}"
    if session is not None:
        sub_dir = sub_dir / f"ses-{session}" / "func"
    else:
        sub_dir = sub_dir / "func"

    entities: list[str] = [f"sub-{subject}"]
    if session is not None:
        entities.append(f"ses-{session}")
    entities.append(f"task-{task}")
    if run is not None:
        entities.append(f"run-{run}")
    entities.append("desc-confounds_timeseries")

    fname = "_".join(entities) + ".tsv"
    return sub_dir / fname


def mask_path(
    fmriprep_dir: Path,
    subject: str,
    task: str,
    *,
    session: str | None = None,
    run: str | int | None = None,
    space: str = DEFAULT_SPACE,
    resolution: str = DEFAULT_RESOLUTION,
) -> Path:
    """Return the expected path to a fMRIPrep brain mask NIfTI.

    Parameters
    ----------
    fmriprep_dir:
        Root of the fMRIPrep derivatives dataset.
    subject:
        Subject label without the ``sub-`` prefix.
    task:
        BIDS task label.
    session:
        BIDS session label without ``ses-`` prefix.
    run:
        Run index or label; omitted when ``None``.
    space:
        fMRIPrep output space template.
    resolution:
        Template resolution label.

    Returns
    -------
    Path
        Full path to the expected brain mask NIfTI file.
    """
    sub_dir = fmriprep_dir / f"sub-{subject}"
    if session is not None:
        sub_dir = sub_dir / f"ses-{session}" / "func"
    else:
        sub_dir = sub_dir / "func"

    entities: list[str] = [f"sub-{subject}"]
    if session is not None:
        entities.append(f"ses-{session}")
    entities.append(f"task-{task}")
    if run is not None:
        entities.append(f"run-{run}")
    if space is not None:
        entities.append(f"space-{space}")
    if resolution is not None:
        entities.append(f"res-{resolution}")
    entities.append("desc-brain_mask")

    fname = "_".join(entities) + ".nii.gz"
    return sub_dir / fname


def events_path(
    bids_dir: Path,
    subject: str,
    task: str,
    *,
    session: str | None = None,
    run: str | int | None = None,
) -> Path:
    """Return the expected path to a BIDS events TSV file in the raw dataset.

    Parameters
    ----------
    bids_dir:
        Root of the raw BIDS dataset.
    subject:
        Subject label without the ``sub-`` prefix.
    task:
        BIDS task label.
    session:
        BIDS session label without ``ses-`` prefix.
    run:
        Run index or label; omitted when ``None``.

    Returns
    -------
    Path
        Full path to the expected BIDS events TSV.
    """
    sub_dir = bids_dir / f"sub-{subject}"
    if session is not None:
        sub_dir = sub_dir / f"ses-{session}" / "func"
    else:
        sub_dir = sub_dir / "func"

    entities: list[str] = [f"sub-{subject}"]
    if session is not None:
        entities.append(f"ses-{session}")
    entities.append(f"task-{task}")
    if run is not None:
        entities.append(f"run-{int(run):02d}")
    entities.append("events")

    fname = "_".join(entities) + ".tsv"
    return sub_dir / fname


# ---------------------------------------------------------------------------
# BIDS entity discovery
# ---------------------------------------------------------------------------


def get_subjects(directory: Path) -> list[str]:
    """Return sorted list of subject labels found under *directory*.

    Scans for ``sub-*`` top-level directories and returns the label without
    the ``sub-`` prefix.

    Parameters
    ----------
    directory:
        A BIDS root directory (raw or derivative).

    Returns
    -------
    list[str]
        Sorted subject labels (e.g. ``["01", "02", "03"]``).
    """
    subjects = sorted(
        p.name[4:]  # strip "sub-"
        for p in directory.iterdir()
        if p.is_dir() and p.name.startswith("sub-")
    )
    return subjects


def get_sessions(directory: Path, subject: str) -> list[str]:
    """Return sorted list of session labels for *subject* under *directory*.

    Parameters
    ----------
    directory:
        A BIDS root directory (raw or derivative).
    subject:
        Subject label without the ``sub-`` prefix.

    Returns
    -------
    list[str]
        Sorted session labels (e.g. ``["001", "002"]``).  Returns ``[]``
        when no ``ses-*`` subdirectories exist (session-less datasets).
    """
    sub_dir = directory / f"sub-{subject}"
    if not sub_dir.exists():
        return []
    sessions = sorted(
        p.name[4:]  # strip "ses-"
        for p in sub_dir.iterdir()
        if p.is_dir() and p.name.startswith("ses-")
    )
    return sessions


def get_bold_runs(
    fmriprep_dir: Path,
    subject: str,
    task: str,
    *,
    session: str | None = None,
    space: str = DEFAULT_SPACE,
    resolution: str = DEFAULT_RESOLUTION,
) -> list[str | None]:
    """Return sorted list of run labels for which a preproc BOLD file exists.

    When no ``run-`` entity is present in the filenames, returns ``[None]``
    to indicate a single run without an explicit run label.

    Parameters
    ----------
    fmriprep_dir:
        Root of the fMRIPrep derivatives dataset.
    subject:
        Subject label without ``sub-`` prefix.
    task:
        BIDS task label.
    session:
        BIDS session label without ``ses-`` prefix.
    space:
        fMRIPrep output space template.
    resolution:
        Template resolution label.

    Returns
    -------
    list[str | None]
        Run labels (without ``run-`` prefix) or ``[None]`` for a single
        unlabelled run.
    """
    sub_dir = fmriprep_dir / f"sub-{subject}"
    if session is not None:
        func_dir = sub_dir / f"ses-{session}" / "func"
    else:
        func_dir = sub_dir / "func"

    if not func_dir.exists():
        return []

    pattern_parts = [
        f"sub-{subject}",
        f"_ses-{session}" if session else "",
        f"_task-{task}*",
    ]
    if space is not None:
        pattern_parts.append(f"_space-{space}")
    if resolution is not None:
        pattern_parts.append(f"_res-{resolution}")
    pattern_parts.append("_desc-preproc_bold.nii.gz")
    
    pattern = "".join(pattern_parts)
    bold_files = sorted(func_dir.glob(pattern))
    runs: list[str | None] = []
    for f in bold_files:
        # Extract run entity from filename
        stem = f.name
        run: str | None = None
        for entity in stem.split("_"):
            if entity.startswith("run-"):
                run = entity[4:]
                break
        runs.append(run)

    return runs if runs else []


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_events_tsv(path: Path) -> pd.DataFrame:
    """Load and validate a BIDS events TSV file.

    BIDS events files must contain at minimum ``onset`` and ``duration``
    columns.  Additional columns (e.g. ``trial_type``, ``stim_file``) are
    preserved.

    Parameters
    ----------
    path:
        Path to the ``*_events.tsv`` file.

    Returns
    -------
    pd.DataFrame
        DataFrame with at least ``onset`` and ``duration`` float columns.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If required BIDS columns are missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"BIDS events file not found: {path}")
    df = pd.read_csv(path, sep="\t")
    required = {"onset", "duration"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"BIDS events file {path} is missing required columns: {missing}"
        )
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    return df


def load_confounds_tsv(
    path: Path,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load a fMRIPrep confounds timeseries TSV.

    Parameters
    ----------
    path:
        Path to the ``*_desc-confounds_timeseries.tsv`` file.
    columns:
        If provided, only these column names are returned.  ``None`` returns
        all columns.  Columns not present in the file are silently ignored.

    Returns
    -------
    pd.DataFrame
        Confounds dataframe, index = volume number.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Confounds file not found: {path}")
    df = pd.read_csv(path, sep="\t")
    if columns is not None:
        keep = [c for c in columns if c in df.columns]
        df = df[keep]
    return df


def iter_bids_runs(
    fmriprep_dir: Path,
    bids_dir: Path | None,
    task: str,
    *,
    subjects: list[str] | None = None,
    space: str = DEFAULT_SPACE,
    resolution: str = DEFAULT_RESOLUTION,
) -> Iterator[dict[str, Any]]:
    """Iterate over all available (subject, session, run) triples in *fmriprep_dir*.

    Only triples for which a preprocessed BOLD file actually exists on disk
    are yielded.  This is the primary iterator used by ``iter_timelines()``
    in all study classes.

    Parameters
    ----------
    fmriprep_dir:
        Root of the fMRIPrep derivatives dataset.
    bids_dir:
        Root of the raw BIDS dataset.  Used for session discovery when the
        fmriprep layout does not include explicit ``ses-*`` directories.
        Pass ``None`` to skip cross-referencing with the raw BIDS layout.
    task:
        BIDS task label.
    subjects:
        Restrict iteration to these subject labels (without ``sub-`` prefix).
        Defaults to all subjects found in *fmriprep_dir*.
    space:
        fMRIPrep output space template.
    resolution:
        Template resolution label.

    Yields
    ------
    dict
        Keys: ``subject`` (str), ``session`` (str | None), ``run`` (str | None),
        ``task`` (str).
    """
    available_subjects = get_subjects(fmriprep_dir)
    if subjects is not None:
        available_subjects = [s for s in available_subjects if s in subjects]

    for sub in available_subjects:
        sessions = get_sessions(fmriprep_dir, sub)
        if not sessions:
            # session-less dataset
            runs = get_bold_runs(
                fmriprep_dir, sub, task, session=None,
                space=space, resolution=resolution
            )
            for run in runs:
                yield dict(subject=sub, session=None, run=run, task=task)
        else:
            for ses in sessions:
                runs = get_bold_runs(
                    fmriprep_dir, sub, task, session=ses,
                    space=space, resolution=resolution
                )
                for run in runs:
                    yield dict(subject=sub, session=ses, run=run, task=task)


def load_bold_masked(
    bold_path_: Path,
    mask_path_: Path | None = None,
) -> "np.ndarray":  # type: ignore[type-arg]
    """Load a BOLD NIfTI and optionally apply a brain mask.

    When *mask_path_* is provided and exists, the function returns a 2-D
    array of shape ``(n_voxels_in_mask, n_volumes)``.  Without masking it
    returns a flattened ``(n_voxels, n_volumes)`` array.

    Parameters
    ----------
    bold_path_:
        Path to the preprocessed BOLD NIfTI file.
    mask_path_:
        Path to a binary brain mask NIfTI.  Shape must match the spatial
        dimensions of *bold_path_*.  When ``None`` or non-existent, no
        masking is applied.

    Returns
    -------
    np.ndarray
        2-D float32 array of shape ``(n_voxels, n_volumes)``.

    Raises
    ------
    FileNotFoundError
        If *bold_path_* does not exist on disk (i.e. DataLad content has not
        been fetched yet).
    """
    import nibabel as nib

    if not bold_path_.exists():
        raise FileNotFoundError(
            f"BOLD file not found (run datalad get or study.download()): {bold_path_}"
        )

    img = nib.load(str(bold_path_))
    data: np.ndarray = np.asarray(img.dataobj, dtype=np.float32)

    # data shape: (X, Y, Z, T)
    n_vols = data.shape[-1]
    flat = data.reshape(-1, n_vols)  # (n_voxels, T)

    if mask_path_ is not None and mask_path_.exists():
        mask_img = nib.load(str(mask_path_))
        mask: np.ndarray = np.asarray(mask_img.dataobj, dtype=bool).ravel()
        flat = flat[mask]

    return flat
