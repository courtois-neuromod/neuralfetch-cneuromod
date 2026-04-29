"""Functional localizer (fLoc) fMRI dataset.

Six subjects completed a standard functional localizer (fLoc) paradigm designed to
identify regions selective for faces, bodies, objects, words, and places.  Images
from five high-level visual categories are displayed in mini-blocks.

References
----------
* CNeuroMod documentation: https://docs.cneuromod.ca/en/latest/DATASETS.html#floc
* DataLad BIDS repo: https://github.com/courtois-neuromod/floc
* DataLad fMRIPrep repo: https://github.com/courtois-neuromod/floc.fmriprep
* fLoc paradigm: https://github.com/VPNL/fLoc
"""

from __future__ import annotations

import typing as tp

import pandas as pd

from neuralfetch_cneuromod.base import CNeuroModStudy
from neuralfetch_cneuromod._utils import events_path, load_events_tsv


#: fLoc visual category labels used as trial_type values in the BIDS events TSV.
FLOC_CATEGORIES = frozenset(
    {"adult", "child", "body", "limb", "car", "instrument", "house", "corridor",
     "word", "number", "scrambled"}
)


class Floc(CNeuroModStudy):
    """Courtois NeuroMod — functional localizer (*fLoc*) dataset.

    Six subjects underwent the fLoc paradigm (5 visual categories, mini-block
    design) during 3T fMRI across multiple sessions.

    Parameters
    ----------
    path:
        Root data directory.  Resolves ``{path}/Floc/bids`` and
        ``{path}/Floc/fmriprep``.
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
    The BIDS events TSV ``trial_type`` column contains category labels such as
    ``"adult"``, ``"child"``, ``"body"``, ``"limb"``, ``"car"``,
    ``"instrument"``, ``"house"``, ``"corridor"``, ``"word"``, ``"number"``,
    and ``"scrambled"``.

    Examples
    --------
    >>> study = Floc(path="/data/cneuromod")
    >>> events = study.run()
    """

    TASK: tp.ClassVar[str] = "floc"
    BIDS_REPO: tp.ClassVar[str] = "floc"
    FMRIPREP_REPO: tp.ClassVar[str] = "floc.fmriprep"

    dataset_name: tp.ClassVar[str] = "CNeuroMod Floc"
    description: tp.ClassVar[str] = (
        "Six subjects completing the fLoc visual localizer paradigm "
        "(faces, bodies, objects, words, places) during 3T fMRI."
    )
    bibtex: tp.ClassVar[str] = CNeuroModStudy.bibtex

    def _load_stimulus_events(
        self, timeline: dict[str, tp.Any]
    ) -> pd.DataFrame:
        """Load fLoc image events with category labels and image file paths.

        Parameters
        ----------
        timeline:
            Timeline dict with ``subject``, ``session``, ``run``, ``task``.

        Returns
        -------
        pd.DataFrame
            Events table with ``type``, ``start``, ``duration``, ``category``,
            and ``filepath`` (absolute path to the stimulus image) columns.
        """
        if not self._bids_dir.exists():
            return pd.DataFrame()

        sub = timeline["subject"]
        ses = timeline.get("session")
        run = timeline.get("run")
        task = timeline.get("task", self.TASK)

        ep = events_path(self._bids_dir, sub, task, session=ses, run=run)
        if not ep.exists():
            return pd.DataFrame()

        bids_events = load_events_tsv(ep)
        stimuli_dir = self._bids_dir / "stimuli"

        rows = []
        for _, row in bids_events.iterrows():
            category = str(row.get("trial_type", "unknown"))
            event: dict[str, tp.Any] = {
                "type": "Image",
                "start": float(row["onset"]),
                "duration": float(row["duration"]),
                "category": category,
            }
            stim_file = row.get("stim_file")
            if stim_file and isinstance(stim_file, str):
                event["filepath"] = str(stimuli_dir / stim_file)
            rows.append(event)

        return pd.DataFrame(rows) if rows else pd.DataFrame()
