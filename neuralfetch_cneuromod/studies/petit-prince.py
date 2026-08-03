"""Le Petit Prince audibook listening fMRI dataset.

Five subjects (sub-01, sub-02, sub-03, sub-05 and sub-06) listened to an 
audiobook version of Le Petit Prince (1943), narrated both in English and 
French by a female-sounding voice, while undergoing 3T fMRI. 

Audio stimuli (.wav) were obtained from:
Jixing Li, John Hale, and Christophe Pallier (2022). Le Petit Prince: A multilingual 
fMRI corpus using ecological stimuli. OpenNeuro. [Dataset] doi: doi:10.18112/openneuro.ds003643.v2.0.0  

The listening task was split into 18 fMRI runs (9 runs in each language) of ~9min to ~13min
administered across multiple sessions for each subject. 

# TODO: petit-prince.stimuli w hyperlinks to external clone of open neuro repo?

References
----------
* DataLad BIDS repo: https://github.com/courtois-neuromod/petit-prince
* DataLad fMRIPrep repo: https://github.com/courtois-neuromod/petit-prince.fmriprep
* DataLad timeseries repo: https://github.com/courtois-neuromod/petit-prince.timeseries
* TODO DataLad stimuli repo: https://github.com/courtois-neuromod/petit-prince.stimuli
* DataLad transcripts repo: https://github.com/courtois-neuromod/petit-prince.annotations
"""

from __future__ import annotations

import typing as tp

import pandas as pd

from neuralfetch_cneuromod.base import CNeuroModStudy
#from neuralfetch_cneuromod._utils import events_path, load_events_tsv


class PetitPrince(CNeuroModStudy):
    """Courtois NeuroMod — *Petit Prince* audiobook listening dataset (EN and FR).

    Five subjects listened to an audiobook verion of Le Petit Prince (1943) in French and 
    then in English during 3T fMRI.  Each BOLD run covers one segment of the book.

    Parameters
    ----------
    path:
        Root data directory.  Resolves ``{path}/petit-prince/bids`` and
        ``{path}/petit-rince/fmriprep``.
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
