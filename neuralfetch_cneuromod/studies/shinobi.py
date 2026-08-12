"""Shinobi video game fMRI dataset.

Four subjects (sub-01, sub-02, sub-04, sub-06) played the 1993 arcade/console game
Shinobi III: Return of the Ninja Master while undergoing 3T fMRI.
Game frame videos and gamepad inputs are stored in the raw BIDS repository.

References
----------
* CNeuroMod documentation: https://docs.cneuromod.ca/latest/datasets/shinobi.html
* DataLad BIDS repo: https://github.com/courtois-neuromod/shinobi
* DataLad fMRIPrep repo: https://github.com/courtois-neuromod/shinobi.fmriprep
"""

from __future__ import annotations

import typing as tp
from pathlib import Path

import pandas as pd

from neuralfetch_cneuromod.base import CNeuroModVideoGameStudy


class Shinobi(CNeuroModStudy):
    """Courtois NeuroMod — *Shinobi* video game fMRI dataset.

    Four subjects (sub-01, sub-02, sub-04, sub-06) played Shinobi III: Return
    of the Ninja Master (1993) while undergoing 3T fMRI.
    Each BOLD run corresponds to a game session with frame-accurate event annotations.

    Parameters
    ----------
    path:
    path:
        Root data directory.  Resolves ``{path}/shinobi/bids`` and
        ``{path}/shinobi/fmriprep``  or ``{path}/shinobi/timeseries``.
    space:
        fMRIPrep output space (default ``"MNI152NLin2009cAsym"``).
    timeseries:
        Define to model pre-extracted, masked, denoised and normalized timeseries, 
        rather than the fMRIPrep BOLD derivatives. Select among ``"cneuromod2026"``, 
        ``"schaefer1000"`` (algonauts 2025 competition), ``"voxel_mni"`` or ``"voxel_native"``.
        (default ``None``) .
    subjects:
        Restrict data loading to a subset of subject labels
        (without ``sub-`` prefix).  ``None`` includes all available subjects.
    datalad_jobs:
        Parallel DataLad download jobs.

    Notes
    -----
    The BIDS events TSV contains game events such as level starts, enemy
    encounters, and player deaths with their frame-accurate onsets.
    
    TODO: implement extraction of game annotations from 
    events.tsv files in bids repo

    Example
    --------
    >>> study = Shinobi(path="path/to/cneuromod.all")
    >>> events = study.run()
    """

    TASK: tp.ClassVar[str] = "shinobi"
    BIDS_REPO: tp.ClassVar[str] = "shinobi"
    FMRIPREP_REPO: tp.ClassVar[str] = "shinobi.fmriprep"

    dataset_name: tp.ClassVar[str] = "CNeuroMod Shinobi"
    description: tp.ClassVar[str] = (
        "Four subjects playing Shinobi III during 3T fMRI."
        "Includes frame-accurate game replays (.mp4)."
    )
    bibtex: tp.ClassVar[str] = CNeuroModStudy.bibtex
