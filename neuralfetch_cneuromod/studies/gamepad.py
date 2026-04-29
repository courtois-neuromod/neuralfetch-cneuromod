"""Gamepad motor task fMRI dataset.

Six subjects performed a gamepad button-pressing motor task during 3T fMRI.
Different fingers and hands were cued, allowing mapping of motor cortex
somatotopy.

References
----------
* CNeuroMod documentation: https://docs.cneuromod.ca/en/latest/DATASETS.html#gamepad
* DataLad BIDS repo: https://github.com/courtois-neuromod/gamepad
* DataLad fMRIPrep repo: https://github.com/courtois-neuromod/gamepad.fmriprep
"""

from __future__ import annotations

import typing as tp

from neuralfetch_cneuromod.base import CNeuroModStudy


class Gamepad(CNeuroModStudy):
    """Courtois NeuroMod — gamepad motor task dataset.

    Six subjects performed button-pressing tasks using a gamepad during 3T
    fMRI, enabling somatotopic mapping of the primary motor cortex.

    Parameters
    ----------
    path:
        Root data directory.  Resolves ``{path}/Gamepad/bids`` and
        ``{path}/Gamepad/fmriprep``.
    space:
        fMRIPrep output space (default ``"MNI152NLin2009cAsym"``).
    resolution:
        MNI resolution label (default ``"2"``).
    subjects:
        Restrict to a subset of subjects.
    datalad_jobs:
        Parallel DataLad download jobs.

    Notes
    -----
    The BIDS events TSV ``trial_type`` encodes which finger/hand was cued
    (e.g. ``"left_index"``, ``"right_thumb"``).

    Examples
    --------
    >>> study = Gamepad(path="/data/cneuromod")
    >>> events = study.run()
    """

    TASK: tp.ClassVar[str] = "gamepad"
    BIDS_REPO: tp.ClassVar[str] = "gamepad"
    FMRIPREP_REPO: tp.ClassVar[str] = "gamepad.fmriprep"

    dataset_name: tp.ClassVar[str] = "CNeuroMod Gamepad"
    description: tp.ClassVar[str] = (
        "Six subjects performing a gamepad motor task during 3T fMRI "
        "for somatotopic cortex mapping."
    )
    bibtex: tp.ClassVar[str] = CNeuroModStudy.bibtex
