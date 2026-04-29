"""Shared pytest fixtures for neuralfetch-cneuromod tests.

Creates minimal synthetic BIDS and fMRIPrep directory trees in a
temporary directory so that tests can run without real data.
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helper: write a tiny synthetic NIfTI file
# ---------------------------------------------------------------------------

def _write_nifti(path: Path, shape: tuple[int, ...] = (10, 10, 10, 20)) -> None:
    """Write a minimal NIfTI file filled with random float32 data.

    Parameters
    ----------
    path:
        Output file path (created along with any missing parent directories).
    shape:
        Array shape ``(X, Y, Z, T)`` for the NIfTI image.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.random.rand(*shape).astype(np.float32)

    zooms = [2.0, 2.0, 2.0]
    if len(shape) == 4:
        # Use 1 s TR (pixdim[4] = 1.5)
        zooms.append(1.5)
    img = nib.Nifti1Image(data, np.eye(4))
    img.header.set_zooms(zooms)
    nib.save(img, str(path))


def _write_events_tsv(path: Path, n_events: int = 5) -> None:
    """Write a minimal BIDS events TSV file.

    Parameters
    ----------
    path:
        Output ``*_events.tsv`` path.
    n_events:
        Number of event rows to generate.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["onset\tduration\ttrial_type\tstim_file"]
    for i in range(n_events):
        lines.append(f"{i * 2.0:.1f}\t1.0\tStimulus\tstim_{i:03d}.png")
    path.write_text("\n".join(lines) + "\n")


def _write_confounds_tsv(path: Path, n_volumes: int = 20) -> None:
    """Write a minimal fMRIPrep confounds timeseries TSV.

    Parameters
    ----------
    path:
        Output ``*_desc-confounds_timeseries.tsv`` path.
    n_volumes:
        Number of rows (time points).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "trans_x\ttrans_y\ttrans_z\trot_x\trot_y\trot_z\tglobal_signal"
    lines = [header]
    for _ in range(n_volumes):
        row = "\t".join(f"{v:.6f}" for v in np.random.randn(7))
        lines.append(row)
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cneuromod_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped temporary CNeuroMod root directory.

    Layout (one subject, one session, two runs for the ``friends`` task)::

        tmp/
        └── Friends/
            ├── bids/
            │   ├── dataset_description.json
            │   └── sub-01/ses-001/func/
            │       ├── sub-01_ses-001_task-friends_run-1_events.tsv
            │       └── sub-01_ses-001_task-friends_run-2_events.tsv
            └── fmriprep/
                └── sub-01/ses-001/func/
                    ├── sub-01_ses-001_task-friends_run-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
                    ├── sub-01_ses-001_task-friends_run-1_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz
                    ├── sub-01_ses-001_task-friends_run-1_desc-confounds_timeseries.tsv
                    ├── sub-01_ses-001_task-friends_run-2_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
                    └── sub-01_ses-001_task-friends_run-2_desc-confounds_timeseries.tsv

    Returns
    -------
    Path
        Root directory used as ``path`` argument for study classes.
    """
    root = tmp_path_factory.mktemp("cneuromod_root")

    SUBJECT = "01"
    SESSION = "001"
    TASK = "friends"
    SPACE = "MNI152NLin2009cAsym"
    RES = "2"
    N_VOLS = 20

    for run in (1, 2):
        # fMRIPrep BOLD
        bold = (
            root / "Friends" / "fmriprep"
            / f"sub-{SUBJECT}" / f"ses-{SESSION}" / "func"
            / f"sub-{SUBJECT}_ses-{SESSION}_task-{TASK}_run-{run}"
              f"_space-{SPACE}_res-{RES}_desc-preproc_bold.nii.gz"
        )
        _write_nifti(bold, shape=(10, 10, 10, N_VOLS))

        # Brain mask (run-1 only — mask_path falls back gracefully)
        if run == 1:
            mask = bold.parent / bold.name.replace(
                "desc-preproc_bold", "desc-brain_mask"
            )
            _write_nifti(mask, shape=(10, 10, 10))

        # Confounds TSV
        conf = bold.parent / (
            f"sub-{SUBJECT}_ses-{SESSION}_task-{TASK}_run-{run}"
            "_desc-confounds_timeseries.tsv"
        )
        _write_confounds_tsv(conf, n_volumes=N_VOLS)

        # BIDS events TSV
        events = (
            root / "Friends" / "bids"
            / f"sub-{SUBJECT}" / f"ses-{SESSION}" / "func"
            / f"sub-{SUBJECT}_ses-{SESSION}_task-{TASK}_run-{run}_events.tsv"
        )
        _write_events_tsv(events)

    # Minimal dataset_description.json (required by BIDS validators)
    desc = root / "Friends" / "bids" / "dataset_description.json"
    desc.parent.mkdir(parents=True, exist_ok=True)
    desc.write_text(
        json.dumps({"Name": "Friends-test", "BIDSVersion": "1.7.0"})
    )

    return root
