"""HCP test-retest (hcptrt) fMRI dataset.

Six subjects underwent a subset of the Human Connectome Project (HCP) task battery
in a test-retest design during 3T fMRI.  Tasks include working memory, motor,
language, and social cognition paradigms.

References
----------
* CNeuroMod documentation: https://docs.cneuromod.ca/en/latest/DATASETS.html#hcptrt
* DataLad BIDS repo: https://github.com/courtois-neuromod/hcptrt
* DataLad fMRIPrep repo: https://github.com/courtois-neuromod/hcptrt.fmriprep
* HCP protocols: https://www.humanconnectome.org/
"""

from __future__ import annotations

import typing as tp

from neuralfetch_cneuromod.base import CNeuroModStudy


class HcpTrt(CNeuroModStudy):
    """Courtois NeuroMod — HCP test-retest (*hcptrt*) dataset.

    Six subjects performed a subset of the HCP task battery (working memory,
    motor, language, social cognition) in a test-retest design during 3T fMRI.

    Parameters
    ----------
    path:
        Root data directory.  Resolves ``{path}/HcpTrt/bids`` and
        ``{path}/HcpTrt/fmriprep``.
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
    The hcptrt dataset contains multiple tasks within the same session.
    The default ``TASK`` label is ``"hcptrt"`` which covers all HCP sub-tasks.
    The BIDS events TSV ``trial_type`` column encodes the specific HCP condition
    (e.g. ``"2-back"``, ``"0-back"``, ``"left_hand"``).

    Examples
    --------
    >>> study = HcpTrt(path="/data/cneuromod")
    >>> events = study.run()
    """

    TASK: tp.ClassVar[str] = "hcptrt"
    BIDS_REPO: tp.ClassVar[str] = "hcptrt"
    FMRIPREP_REPO: tp.ClassVar[str] = "hcptrt.fmriprep"

    dataset_name: tp.ClassVar[str] = "CNeuroMod HcpTrt"
    description: tp.ClassVar[str] = (
        "Six subjects performing the HCP task battery (working memory, motor, "
        "language, social cognition) in a test-retest design during 3T fMRI."
    )
    bibtex: tp.ClassVar[str] = CNeuroModStudy.bibtex
