"""Friends TV show fMRI dataset.

Six subjects watched all episodes of Friends seasons 1-6 (approximately 50 hours
of audio-visual stimulation) while being scanned at 3T.  Each episode is split
into multiple BOLD runs.  Stimuli are provided as video clip files in the raw BIDS
repository.

References
----------
* CNeuroMod documentation: https://docs.cneuromod.ca/en/latest/DATASETS.html#friends
* DataLad BIDS repo: https://github.com/courtois-neuromod/friends
* DataLad fMRIPrep repo: https://github.com/courtois-neuromod/friends.fmriprep
"""

from __future__ import annotations

import typing as tp
from pathlib import Path

import pandas as pd

from neuralfetch_cneuromod.base import CNeuroModStudy
#from neuralfetch_cneuromod._utils import events_path, load_events_tsv


class Friends(CNeuroModStudy):
    """Courtois NeuroMod — *Friends* TV show fMRI dataset.

    Six subjects (01, 02, 03, 04, 05, 06) watched Friends seasons 1-6 across
    hundreds of scanning sessions.  Each BOLD run corresponds to one ~8-minute
    segment of a Friends episode.

    Parameters
    ----------
    path:
        Root data directory.  The study resolves ``{path}/Friends/bids`` and
        ``{path}/Friends/fmriprep``.
    space:
        fMRIPrep output space (default ``"MNI152NLin2009cAsym"``).
    resolution:
        MNI resolution label (default ``"2"``).
    subjects:
        Restrict to a subset of subjects (labels without ``sub-`` prefix).
    datalad_jobs:
        Parallel DataLad download jobs.

    Notes
    -----
    * Stimuli (mkv video clips) are stored in the BIDS ``stimuli/`` directory.
      Call ``study.bids_dir / "stimuli"`` to locate them after running
      ``datalad get``.
    * The BIDS events TSV contains ``onset``, ``duration``, and ``trial_type``
      columns that identify scene boundaries.

    Examples
    --------
    >>> study = Friends(path="/data/cneuromod")
    >>> print(study.study_summary().head())
    >>> events = study.run()
    """

    TASK: tp.ClassVar[str] = "friends"
    BIDS_REPO: tp.ClassVar[str] = "friends"
    FMRIPREP_REPO: tp.ClassVar[str] = "friends.fmriprep"

    dataset_name: tp.ClassVar[str] = "CNeuroMod Friends"
    description: tp.ClassVar[str] = (
        "Six subjects watching Friends TV show seasons 1-6 (≈50 h) "
        "during 3T fMRI acquisition."
    )
    bibtex: tp.ClassVar[str] = CNeuroModStudy.bibtex

    def _load_stimulus_events(
        self, timeline: dict[str, tp.Any]
    ) -> pd.DataFrame:
        """Load Friends events and attach the video stimulus file path.

        For each BOLD run the corresponding BIDS events TSV is read.  The
        ``stim_file`` column (when present) is converted to an absolute path
        relative to the BIDS stimuli directory.

        Parameters
        ----------
        timeline:
            Timeline dict with ``subject``, ``session``, ``run``, ``task``.

        Returns
        -------
        pd.DataFrame
            Events table with columns ``type``, ``start``, ``duration``,
            and ``filepath`` (absolute path to the video clip, when available).
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
                "type": str(row.get("trial_type", "Stimulus")),
                "start": float(row["onset"]),
                "duration": float(row["duration"]),
            }
            stim_file = row.get("stim_file")
            if stim_file and isinstance(stim_file, str):
                candidate = stimuli_dir / stim_file
                event["filepath"] = str(candidate)
            rows.append(event)

        return pd.DataFrame(rows) if rows else pd.DataFrame()
