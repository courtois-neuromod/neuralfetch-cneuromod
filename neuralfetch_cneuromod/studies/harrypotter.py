"""Harry Potter audiobook listening fMRI dataset.

Six subjects listened to the Harry Potter and the Philosopher's Stone audiobook
(chapters 1-9) while undergoing 3T fMRI.  Stimuli are audio files (wav/mp3)
stored in the raw BIDS repository.

References
----------
* CNeuroMod documentation: https://docs.cneuromod.ca/en/latest/DATASETS.html#harrypotter
* DataLad BIDS repo: https://github.com/courtois-neuromod/harrypotter
* DataLad fMRIPrep repo: https://github.com/courtois-neuromod/harrypotter.fmriprep
"""

from __future__ import annotations

import typing as tp

import pandas as pd

from neuralfetch_cneuromod.base import CNeuroModStudy
from neuralfetch_cneuromod._utils import events_path, load_events_tsv


class HarryPotter(CNeuroModStudy):
    """Courtois NeuroMod — *Harry Potter* audiobook listening dataset.

    Six subjects listened to Harry Potter and the Philosopher's Stone
    (chapters 1-9) during 3T fMRI.  Each BOLD run covers one chapter or
    chapter segment.

    Parameters
    ----------
    path:
        Root data directory.  Resolves ``{path}/HarryPotter/bids`` and
        ``{path}/HarryPotter/fmriprep``.
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
    The BIDS events TSV contains word-level onsets derived from forced
    alignment of the audiobook text with the audio stimulus.

    Examples
    --------
    >>> study = HarryPotter(path="/data/cneuromod")
    >>> events = study.run()
    """

    TASK: tp.ClassVar[str] = "harrypotter"
    BIDS_REPO: tp.ClassVar[str] = "harrypotter"
    FMRIPREP_REPO: tp.ClassVar[str] = "harrypotter.fmriprep"

    dataset_name: tp.ClassVar[str] = "CNeuroMod HarryPotter"
    description: tp.ClassVar[str] = (
        "Six subjects listening to the Harry Potter audiobook (chapters 1-9) "
        "during 3T fMRI."
    )
    bibtex: tp.ClassVar[str] = CNeuroModStudy.bibtex

    def _load_stimulus_events(
        self, timeline: dict[str, tp.Any]
    ) -> pd.DataFrame:
        """Load Harry Potter word-level events with audio file paths.

        Parameters
        ----------
        timeline:
            Timeline dict with ``subject``, ``session``, ``run``, ``task``.

        Returns
        -------
        pd.DataFrame
            Events table with ``type``, ``start``, ``duration``, and
            optionally ``word``, ``filepath`` columns.
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
                "type": "Text",
                "start": float(row["onset"]),
                "duration": float(row["duration"]),
                "language": "english",
                "modality": "read"
            }
            # Word text annotation
            for col in ("word", "trial_type"):
                if col in bids_events.columns:
                    event["text"] = str(row[col])
                    break
            rows.append(event)

        return pd.DataFrame(rows) if rows else pd.DataFrame()
