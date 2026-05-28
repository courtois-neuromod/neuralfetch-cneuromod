"""Shinobi video game fMRI dataset.

Six subjects played the Shinobi arcade/console game during 3T fMRI.
Game frame videos and gamepad inputs are stored in the raw BIDS repository.

References
----------
* DataLad BIDS repo: https://github.com/courtois-neuromod/shinobi
* DataLad fMRIPrep repo: https://github.com/courtois-neuromod/shinobi.fmriprep
"""

from __future__ import annotations

import typing as tp

import pandas as pd

from neuralfetch_cneuromod.base import CNeuroModStudy
#from neuralfetch_cneuromod._utils import events_path, load_events_tsv


class Shinobi(CNeuroModStudy):
    """Courtois NeuroMod — *Shinobi* video game fMRI dataset.

    Six subjects played the Shinobi game during 3T fMRI.  Each BOLD run
    corresponds to a game session with frame-accurate event annotations.

    Parameters
    ----------
    path:
        Root data directory.  Resolves ``{path}/Shinobi/bids`` and
        ``{path}/Shinobi/fmriprep``.
    space:
        fMRIPrep output space (default ``"MNI152NLin2009cAsym"``).
    resolution:
        MNI resolution label (default ``"2"``).
    subjects:
        Restrict to a subset of subjects.
    datalad_jobs:
        Parallel DataLad download jobs.

    Examples
    --------
    >>> study = Shinobi(path="/data/cneuromod")
    >>> events = study.run()
    """

    TASK: tp.ClassVar[str] = "shinobi"
    BIDS_REPO: tp.ClassVar[str] = "shinobi"
    FMRIPREP_REPO: tp.ClassVar[str] = "shinobi.fmriprep"

    dataset_name: tp.ClassVar[str] = "CNeuroMod Shinobi"
    description: tp.ClassVar[str] = (
        "Six subjects playing the Shinobi video game during 3T fMRI."
    )
    bibtex: tp.ClassVar[str] = CNeuroModStudy.bibtex

    def _load_stimulus_events(
        self, timeline: dict[str, tp.Any]
    ) -> pd.DataFrame:
        """Load Shinobi game events.

        Parameters
        ----------
        timeline:
            Timeline dict with ``subject``, ``session``, ``run``, ``task``.

        Returns
        -------
        pd.DataFrame
            Events table with ``type``, ``start``, ``duration`` columns.
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
            event: dict[str, tp.Any] = {
                "type": str(row.get("trial_type", "GameEvent")),
                "start": float(row["onset"]),
                "duration": float(row["duration"]),
            }
            stim_file = row.get("stim_file")
            if stim_file and isinstance(stim_file, str):
                event["filepath"] = str(stimuli_dir / stim_file)
            rows.append(event)

        return pd.DataFrame(rows) if rows else pd.DataFrame()
