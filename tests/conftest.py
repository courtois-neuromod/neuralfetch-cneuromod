"""Shared pytest fixtures for neuralfetch-cneuromod tests.

Creates minimal synthetic BIDS and fMRIPrep directory trees in a
temporary directory so that tests can run without real data.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
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
        └── friends/
            ├── bids/
            │   ├── dataset_description.json
            │   └── sub-01/ses-001/func/
            │       ├── sub-01_ses-001_task-s01e01a_events.tsv
            │       └── sub-01_ses-001_task-s01e01b_events.tsv
            ├── timeseries/
            │   └── timeseries/cneuromod26/sub-01/
            │       └── sub-01_task-friends_space-MNI152NLin2009cAsym_atlas-cneuromod26_desc-1134Parcels_timeseries.h5
            └── fmriprep/
                └── sub-01/ses-001/func/
                    ├── sub-01_ses-001_task-s01e01a_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
                    ├── sub-01_ses-001_task-s01e01a_desc-confounds_timeseries.tsv
                    ├── sub-01_ses-001_task-s01e01b_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
                    └── sub-01_ses-001_task-s01e01b_desc-confounds_timeseries.tsv

    Returns
    -------
    Path
        Root directory used as ``path`` argument for study classes.
    """
    root = tmp_path_factory.mktemp("cneuromod.all")

    SUBJECT = "01"
    SESSION = "001"
    DSET = "friends"
    # Base for Fiends episodes
    TASK = "s01e01{episode_seg}"    
    SPACE = "MNI152NLin2009cAsym"
    TIMESERIES = "cneuromod2026"
    ATLAS = "cneuromod26"
    DESC = "1134Parcels"
    N_VOLS = 20


    # Timeseries nested .HDF5
    tseries = (
        root / "friends" / "timeseries" / "timeseries"
        / TIMESERIES / f"sub-{SUBJECT}" 
        / f"sub-{SUBJECT}_task-{DSET}_space-{SPACE}"
          f"_atlas-{ATLAS}_desc-{DESC}_timeseries.h5"
    )
    tseries.parent.mkdir(parents=True)

    for epi_seg in ("a", "b"):
        # fMRIPrep BOLD
        RUN = TASK.format(episode_seg=epi_seg)

        flag = "a" if tseries.exists() else "w"
        with h5py.File(tseries, flag) as f:
            ses_group = f.create_group(
                f"ses-{SESSION}"
            ) if f"ses-{SESSION}" not in f else f[f"ses-{SESSION}"]
            ses_group.create_dataset(
                f"ses-{SESSION}_task-{RUN}_timeseries",
                data=np.zeros((10, 10))
            )

        bold = (
            root / "friends" / "fmriprep"
            / f"sub-{SUBJECT}" / f"ses-{SESSION}" / "func"
            / f"sub-{SUBJECT}_ses-{SESSION}_task-{RUN}"
              f"_space-{SPACE}_desc-preproc_bold.nii.gz"
        )
        _write_nifti(bold, shape=(10, 10, 10, N_VOLS))

        # Brain mask (half episode "a" only)
        if epi_seg == "a":
            mask = bold.parent / bold.name.replace(
                "desc-preproc_bold", "desc-brain_mask"
            )
            # Create a mask with some zeros so masking actually reduces the array
            mask_data = np.zeros((10, 10, 10), dtype=np.float32)
            mask_data[2:8, 2:8, 2:8] = 1.0
            mask_img = nib.Nifti1Image(mask_data, np.eye(4))
            mask_img.header.set_zooms([2.0, 2.0, 2.0])
            nib.save(mask_img, str(mask))

        # Confounds TSV
        conf = bold.parent / (
            f"sub-{SUBJECT}_ses-{SESSION}_task-{RUN}"
            "_desc-confounds_timeseries.tsv"
        )
        _write_confounds_tsv(conf, n_volumes=N_VOLS)

        # BIDS events TSV
        events = (
            root / "friends" / "bids"
            / f"sub-{SUBJECT}" / f"ses-{SESSION}" / "func"
            / f"sub-{SUBJECT}_ses-{SESSION}_task-{RUN}_events.tsv"
        )
        _write_events_tsv(events)

    # Minimal dataset_description.json (required by BIDS validators)
    desc = root / "friends" / "bids" / "dataset_description.json"
    desc.parent.mkdir(parents=True, exist_ok=True)
    desc.write_text(
        json.dumps({"Name": "friends-test", "BIDSVersion": "1.7.0"})
    )

    return root

