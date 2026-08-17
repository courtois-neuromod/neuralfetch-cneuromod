"""Super Mario Bros videogame playing fMRI dataset.

Five subjects (sub-01, sub-02, sub-03, sub-05, sub-06) played Super Mario All-Stars (NES)
while undergoing 3T fMRI. Each session included several runs featuring a mixture of
game levels. 

Levels were mixed pseudo-randomly within and across runs throughout the different sessions.
Participants were given up to three attempts ("lives") to successfully clear a given level.

Stimuli are game frame video files (a.k.a. game replays); gamepad input logs are provided in
the raw BIDS repository.


References
----------
* CNeuroMod documentation: https://docs.cneuromod.ca/latest/datasets/mariostars.html
* DataLad BIDS repo: https://github.com/courtois-neuromod/mariostars
* DataLad fMRIPrep repo: https://github.com/courtois-neuromod/mariostars.fmriprep
* DataLad timeseries repo: https://github.com/courtois-neuromod/mariostars.timeseries
"""

from __future__ import annotations

import typing as tp

from neuralfetch_cneuromod.base import CNeuroModVideoGameStudy


class MarioStars(CNeuroModVideoGameStudy):
    """Courtois NeuroMod — *Super Mario All-Stars* videogaming fMRI dataset.

    Five subjects (sub-01, sub-02, sub-03, sub-05, sub-06) played Super Mario All-Stars (NES)
    while undergoing 3T fMRI. Each BOLD run corresponds to multiple naturalistic gameplays.

    Parameters
    ----------
    path:
        Root data directory.  Resolves ``{path}/mariostars/bids`` and
        ``{path}/mariostars/fmriprep``  or ``{path}/mariostars/timeseries``.
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
    >>> study = MarioStars(path="path/to/cneuromod.all")
    >>> events = study.run()
    """

    TASK: tp.ClassVar[str] = "mariostars"
    BIDS_REPO: tp.ClassVar[str] = "mariostars"
    FMRIPREP_REPO: tp.ClassVar[str] = "mariostars.fmriprep"
    TIMESERIES_REPO: tp.ClassVar[str] = "mariostars.timeseries"

    dataset_name: tp.ClassVar[str] = "CNeuroMod MarioStars"
    description: tp.ClassVar[str] = (
        "Five subjects playing Super Mario All-Stars during 3T fMRI. "
        "Includes frame-accurate game replays (.mp4)."
    )
    bibtex: tp.ClassVar[str] = CNeuroModVideoGameStudy.bibtex
