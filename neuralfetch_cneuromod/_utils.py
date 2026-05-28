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

import glob
import logging
from pathlib import Path
from typing import Any, Iterator, Sequence

import h5py
import numpy as np
import pandas as pd
from datalad import api as dl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BIDS / fMRIPrep path builders
# ---------------------------------------------------------------------------

#: Default MNI152 template identifier used by fMRIPrep.
DEFAULT_SPACE = "MNI152NLin2009cAsym"
#: Default fMRI TR in seconds.
DEFAULT_TR = 1.49
#: Default resolution string used by fMRIPrep.
DEFAULT_RESOLUTION: str | None = "2"
#: Timeseries file name descriptor 
DEFAULT_TIMESERIES = "cneuromod2026"
TSERIES_DESCRIPT = {
    "cneuromod2026": "atlas-cneuromod26_desc-1134Parcels",
    "schaefer1000": "atlas-Schaefer18_desc-1000Parcels7Networks",
    "voxel_mni": "desc-voxelwise",
    "voxel_native": "desc-voxelwise",
}


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
        Sorted session labels (e.g. ``["001", "002"]``).
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
    *,
    session: str | None = None,
    space: str = DEFAULT_SPACE,
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
    session:
        BIDS session label without ``ses-`` prefix.
    space:
        fMRIPrep output space template.

    Returns
    -------
    list[tuple[str, Path, str]]
        (Task run label (without ``task-`` prefix), 
        run number (without ``run-`` prefix) or ``None`` if no run number,
        full run path).
    """
    sub_dir = fmriprep_dir / f"sub-{subject}" / f"ses-{session}" / "func"

    if not sub_dir.exists():
        return []

    pattern_parts = [
        f"sub-{subject}_ses-{session}_task-*",
        f"_space-{space}_desc-preproc_bold.nii.gz",
    ]
    
    pattern = "".join(pattern_parts)
    bold_files = sorted(sub_dir.glob(pattern))
    runs = []
    for f in bold_files:
        # Extract task-run entity from filename
        stem = f.name
        run: str | None = None
        task: str | None = None        
        for entity in stem.split("_"):
            if entity.startswith("task-"):
                task = entity[5:]
            elif entity.startswith("run-"):
                run = entity[4:]
                break
        runs.append((task, run, f))

    return runs if runs else []


# ---------------------------------------------------------------------------
# Datalad get helpers
# ---------------------------------------------------------------------------

def datalad_get_list(
    patterns: list,
    dset_path: str
) -> None:
    """Pulls files selectively from a dataset submodule by passing BIDS glob 
    patterns to ``datalad get``.

    Parameters
    ----------
    patterns:
        List of file patterns to download from a submodule
    dset_path:
        Path to the submodule that contains the files (the 'dataset')

    """
    for pattern in patterns:
        dl_files = sorted(glob.glob(pattern))
        if len(dl_files):
            dl.get(path=dl_files, dataset=dset_path, jobs="auto")


def _set_dir_permissions(path: Path) -> None:
    """Overwrites study._set_dir_permissions() helper function that sets 
    777 permissions recursively on a dataset repository. 
    
    This reset is overly permissive, and it is incompatible with the selective 
    file download implemented with datalad get.
    """
    pass

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

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
    *,
    subjects: list[str] | None = None,
    space: str = DEFAULT_SPACE,
) -> Iterator[dict[str, Any]]:
    """Iterate over all available (subject, session, run) triples in *fmriprep_dir*.

    Only triples for which a preprocessed BOLD file actually exists on disk
    are yielded.  This is the primary iterator used by ``iter_timelines()``
    in all study classes.

    Parameters
    ----------
    fmriprep_dir:
        Root of the fMRIPrep derivatives dataset.
    subjects:
        Restrict iteration to these subject labels (without ``sub-`` prefix).
        Defaults to all subjects found in *fmriprep_dir*.
    space:
        fMRIPrep output space template.

    Yields
    ------
    dict
        Keys: ``subject`` (str), ``file_path`` (Path), ``session`` (str), ``task`` (str),
        ``run`` (str | None).
    """
    available_subjects = get_subjects(fmriprep_dir)
    if subjects is not None:
        available_subjects = [s for s in available_subjects if s in subjects]

    for sub in available_subjects:
        sessions = get_sessions(fmriprep_dir, sub)
        for ses in sessions:
            runs = get_bold_runs(
                fmriprep_dir, sub, 
                session=ses, space=space,
            )
            for task, run, run_path in runs:
                yield dict(subject=sub, file_path=run_path, session=ses, task=task, run=run)


def iter_tseries_runs(
    timeseries_dir: Path,
    *,
    task: str,
    subjects: list[str] | None = None,
    timeseries: str = DEFAULT_TIMESERIES,
    space: str = DEFAULT_SPACE,
) -> Iterator[dict[str, Any]]:
    """Iterate over all available (subject, session, run) triples in nested 
    .h5 timeseries files for specified timeseries in *timeseries_dir*.

    Only triples for which a preprocessed timeseries actually exists
    are yielded.  This is the alternative iterator used by ``iter_timelines()``
    to compile timeseries in all study classes for which they have been extracted.

    Parameters
    ----------
    timeseries_dir:
        Root of the timeseries derivatives dataset.
    task: 
        cneuromod dataset name (e.g., movie10)
    subjects:
        Restrict iteration to these subject labels (without ``sub-`` prefix).
        Defaults to all subjects found in *fmriprep_dir*.
    timeseries:
        name of the pre-extracted, masked, denoised and normalized timeseries
    space:
        fMRIPrep output space template of BOLD data processed into timeseries.

    Yields
    ------
    dict
        Keys: ``subject`` (str), ``file_path`` (Path), ``session`` (str),  ``task`` (None), ``run`` (str).
    """
    available_subjects = get_subjects(Path(
        f"{timeseries_dir}/timeseries/{timeseries}"
    ))
    if subjects is not None:
        available_subjects = [s for s in available_subjects if s in subjects]

    for sub in available_subjects:
        h5_path = Path(
            f"{timeseries_dir}/timeseries/{timeseries}/sub-{sub}/sub-{sub}_task-{task}"
            f"_space-{space}_{TSERIES_DESCRIPT[timeseries]}_timeseries.h5",
        )
        sub_tseries = h5py.File(h5_path, "r")
        sessions = list(sub_tseries.keys())
        for ses in sessions:
            runs = list(sub_tseries[ses].keys())
            for run in runs:
                yield dict(subject=sub, file_path=h5_path, session=ses, task=None, run=run)


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
