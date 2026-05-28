"""Super Mario Bros gameplay fMRI dataset.

Six subjects played Super Mario Bros (NES) and Super Mario Bros 3 during 3T fMRI.
Each session consists of a fixed sequence of game levels.  Stimuli are game frame
video files; gamepad input logs are provided in the raw BIDS repository.

References
----------
* CNeuroMod documentation: https://docs.cneuromod.ca/en/latest/DATASETS.html#mario
* DataLad BIDS repo: https://github.com/courtois-neuromod/mario
* DataLad fMRIPrep repo: https://github.com/courtois-neuromod/mario.fmriprep
"""

from __future__ import annotations

import typing as tp

import pandas as pd

from neuralfetch_cneuromod.base import CNeuroModStudy
#from neuralfetch_cneuromod._utils import events_path, load_events_tsv


class Mario(CNeuroModStudy):
    """Courtois NeuroMod — *Super Mario Bros* gameplay fMRI dataset.

    Six subjects played Super Mario Bros (NES) while undergoing 3T fMRI.
    Each BOLD run corresponds to a fixed game sequence.

    Parameters
    ----------
    path:
        Root data directory.  Resolves ``{path}/Mario/bids`` and
        ``{path}/Mario/fmriprep``.
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
    The BIDS events TSV contains game events such as level starts, enemy
    encounters, and player deaths with their frame-accurate onsets.

    Examples
    --------
    >>> study = Mario(path="/data/cneuromod")
    >>> events = study.run()
    """

    TASK: tp.ClassVar[str] = "mario"
    BIDS_REPO: tp.ClassVar[str] = "mario"
    FMRIPREP_REPO: tp.ClassVar[str] = "mario.fmriprep"

    dataset_name: tp.ClassVar[str] = "CNeuroMod Mario"
    description: tp.ClassVar[str] = (
        "Six subjects playing Super Mario Bros during 3T fMRI. "
        "Includes frame-accurate game event annotations."
    )
    bibtex: tp.ClassVar[str] = CNeuroModStudy.bibtex

    def _load_stimulus_events(
        self, timeline: dict[str, tp.Any]
    ) -> pd.DataFrame:
        """Load Mario game events with frame video file paths.

        Parameters
        ----------
        timeline:
            Timeline dict with ``subject``, ``session``, ``run``, ``task``.

        Returns
        -------
        pd.DataFrame
            Events table with ``type``, ``start``, ``duration``, and optional
            ``filepath`` (game frame video) and ``game_event`` columns.
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
