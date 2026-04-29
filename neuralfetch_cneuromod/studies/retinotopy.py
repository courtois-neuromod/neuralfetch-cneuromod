"""Retinotopy (population receptive field mapping) fMRI dataset.

Six subjects underwent a retinotopy paradigm designed for population receptive
field (pRF) mapping of early visual cortex during 3T fMRI.  Stimuli are
contrast-reversing checkerboard bar apertures sweeping across the visual field.

References
----------
* CNeuroMod documentation: https://docs.cneuromod.ca/en/latest/DATASETS.html#retinotopy
* DataLad BIDS repo: https://github.com/courtois-neuromod/retinotopy
* DataLad fMRIPrep repo: https://github.com/courtois-neuromod/retinotopy.fmriprep
"""

from __future__ import annotations

import typing as tp

from neuralfetch_cneuromod.base import CNeuroModStudy


class Retinotopy(CNeuroModStudy):
    """Courtois NeuroMod — retinotopy / pRF mapping dataset.

    Six subjects viewed travelling-wave bar stimuli for population receptive
    field (pRF) mapping of early visual cortex during 3T fMRI.

    Parameters
    ----------
    path:
        Root data directory.  Resolves ``{path}/Retinotopy/bids`` and
        ``{path}/Retinotopy/fmriprep``.
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
    The retinotopy dataset is typically used with dedicated pRF fitting tools
    (e.g. ``prfpy`` or FreeSurfer ``mri_glmfit``), not with event-based GLMs.
    The events TSV encodes bar pass onset/duration and direction.

    Examples
    --------
    >>> study = Retinotopy(path="/data/cneuromod")
    >>> events = study.run()
    """

    TASK: tp.ClassVar[str] = "retinotopy"
    BIDS_REPO: tp.ClassVar[str] = "retinotopy"
    FMRIPREP_REPO: tp.ClassVar[str] = "retinotopy.fmriprep"

    dataset_name: tp.ClassVar[str] = "CNeuroMod Retinotopy"
    description: tp.ClassVar[str] = (
        "Six subjects performing travelling-wave bar pRF mapping during 3T fMRI."
    )
    bibtex: tp.ClassVar[str] = CNeuroModStudy.bibtex
