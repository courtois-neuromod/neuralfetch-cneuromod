"""Movie10 open-access Hollywood movies fMRI dataset.

Six subjects watched three Hollywood feature films ("The Bourne Supremacy" (2004), 
"The Wolf of Wall Street" (2013), "Hidden Figures" (2016)) and one BBC nature documentary 
("Life : Challenges of life, reptiles and amphibian mammals" (2009)) totalling ~10 hours of 
movie watching while undergoing 3T fMRI.  
Hidden Figure and Life were both visioned twice to support reproducibility analyses.
Each movie was split into multiple ~10-minute BOLD runs.

References
----------
* CNeuroMod documentation: https://docs.cneuromod.ca/en/latest/DATASETS.html#movie10
* DataLad BIDS repo: https://github.com/courtois-neuromod/movie10
* DataLad fMRIPrep repo: https://github.com/courtois-neuromod/movie10.fmriprep
"""

from __future__ import annotations

import typing as tp

import pandas as pd

from neuralfetch_cneuromod.base import CNeuroModStudy
from neuralfetch_cneuromod._utils import events_path, load_events_tsv


class Movie10(CNeuroModStudy):
    """Courtois NeuroMod — *Movie10* movie-watching fMRI dataset.

    Six subjects watched 10 hours of Hollywood movies/BBC documentary while 
    undergoing 3T fMRI. Movie titles include *Life (2009) (shown twice)*,
    *Hidden Figures (2016)*, *The Wolf of Wall Street (2013)*, and 
    *The Bourne Supremacy (2004)*.

    Parameters
    ----------
    path:
        Root data directory.  Resolves ``{path}/movie10/bids`` and
        ``{path}/movie10/fmriprep``.
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
    >>> study = Movie10(path="/data/cneuromod.all")
    >>> events = study.run()
    """

    TASK: tp.ClassVar[str] = "movie10"
    BIDS_REPO: tp.ClassVar[str] = "movie10"
    FMRIPREP_REPO: tp.ClassVar[str] = "movie10.fmriprep"

    dataset_name: tp.ClassVar[str] = "CNeuroMod Movie10"
    description: tp.ClassVar[str] = (
        "Six subjects watching 10 hours of Hollywood movies / BBC documentary during 3T fMRI."
    )
    bibtex: tp.ClassVar[str] = CNeuroModStudy.bibtex

    def _load_stimulus_events(
        self, timeline: dict[str, tp.Any]
    ) -> pd.DataFrame:
        """Load Movie10 stimulus events with video clip file paths.

        Parameters
        ----------
        timeline:
            Timeline dict with ``subject``, ``session``, ``run``, ``task``.

        Returns
        -------
        pd.DataFrame
            Events table with ``type``, ``start``, ``duration``, ``filepath``.
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
                event["filepath"] = str(stimuli_dir / stim_file)
            rows.append(event)

        return pd.DataFrame(rows) if rows else pd.DataFrame()
