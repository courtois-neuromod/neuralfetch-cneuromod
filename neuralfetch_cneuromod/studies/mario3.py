"""Super Mario Bros videogame playing fMRI dataset.

Five subjects (sub-01, sub-02, sub-03, sub-05, sub-06) played Super Mario Bros 3 (NES)
while undergoing 3T fMRI. Each session included several runs featuring a mixture of
game levels. 

Sessions were split into two phases. During the discovery phase, participants played each 
game level in order. Each level had to be cleared at least once before the participant could
play the next level, until they finished the entire game.

During the stable phase, levels were mixed pseudo-randomly within and across runs.
Participants were given up to three attempts ("lives") to successfully clear a given level.

Stimuli are game frame video files (a.k.a. game replays); gamepad input logs are provided in
the raw BIDS repository.


References
----------
* CNeuroMod documentation: https://docs.cneuromod.ca/latest/datasets/mario3.html
* DataLad BIDS repo: https://github.com/courtois-neuromod/mario3
* DataLad fMRIPrep repo: https://github.com/courtois-neuromod/mario3.fmriprep
* DataLad timeseries repo: https://github.com/courtois-neuromod/mario3.timeseries
"""

from __future__ import annotations

import typing as tp

from neuralfetch_cneuromod.base import CNeuroModVideoGameStudy


class Mario3(CNeuroModVideoGameStudy):
    """Courtois NeuroMod — *Super Mario Bros 3* videogaming fMRI dataset.

    Five subjects (sub-01, sub-02, sub-03, sub-05, sub-06) played Super Mario Bros 3 (NES)
    while undergoing 3T fMRI. Each BOLD run corresponds to multiple naturalistic gameplays.

    Parameters
    ----------
    path:
        Root data directory.  Resolves ``{path}/mario3/bids`` and
        ``{path}/mario/fmriprep``  or ``{path}/mario3/timeseries``.
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
    >>> study = Mario3(path="path/to/cneuromod.all")
    >>> events = study.run()
    """

    TASK: tp.ClassVar[str] = "mario3"
    BIDS_REPO: tp.ClassVar[str] = "mario3"
    FMRIPREP_REPO: tp.ClassVar[str] = "mario3.fmriprep"
    TIMESERIES_REPO: tp.ClassVar[str] = "mario3.timeseries"

    dataset_name: tp.ClassVar[str] = "CNeuroMod Mario3"
    description: tp.ClassVar[str] = (
        "Five subjects playing Super Mario Bros 3 during 3T fMRI. "
        "Includes frame-accurate game replays (.mp4)."
    )
    bibtex: tp.ClassVar[str] = CNeuroModStudy.bibtex
