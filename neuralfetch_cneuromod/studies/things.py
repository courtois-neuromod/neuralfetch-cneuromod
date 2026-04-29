"""THINGS object image recognition fMRI dataset.

Six subjects viewed images from the THINGS object concept database
(Hebart et al., 2019) during 3T fMRI.  Each image belongs to one of
1,854 object categories.

References
----------
* CNeuroMod documentation: https://docs.cneuromod.ca/en/latest/DATASETS.html#things
* DataLad BIDS repo: https://github.com/courtois-neuromod/things
* DataLad fMRIPrep repo: https://github.com/courtois-neuromod/things.fmriprep
* THINGS dataset: https://things-initiative.org/
"""

from __future__ import annotations

import typing as tp
from pathlib import Path

import pandas as pd

from neuralfetch_cneuromod.base import CNeuroModStudy
from neuralfetch_cneuromod._utils import events_path, load_events_tsv


class Things(CNeuroModStudy):
    """Courtois NeuroMod — *THINGS* object image fMRI dataset.

    Six subjects performed rapid visual presentation of THINGS images during
    3T fMRI.  Each trial shows one image for a brief duration.

    Parameters
    ----------
    path:
        Root data directory.  Resolves ``{path}/Things/bids`` and
        ``{path}/Things/fmriprep``.
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
    The BIDS events TSV includes ``stim_file`` (relative path to the image),
    ``trial_type`` (object category), and ``response_time`` columns.

    Examples
    --------
    >>> study = Things(path="/data/cneuromod")
    >>> events = study.run()
    """

    TASK: tp.ClassVar[str] = "things"
    BIDS_REPO: tp.ClassVar[str] = "things"
    FMRIPREP_REPO: tp.ClassVar[str] = "things.fmriprep"

    dataset_name: tp.ClassVar[str] = "CNeuroMod Things"
    description: tp.ClassVar[str] = (
        "Six subjects viewing THINGS object images (1,854 categories) "
        "during 3T fMRI."
    )
    bibtex: tp.ClassVar[str] = (
        CNeuroModStudy.bibtex
        + """
    @article{hebart2019things,
        title={THINGS: A database of 1,854 object concepts and more than
        26,000 naturalistic object images},
        author={Hebart, Martin N and Dickter, Adam H and Kidder, Alexis and
        Kwok, Wan Y and Corriveau, Anna and Van Wicklin, Caitlin and
        Baker, Chris I},
        journal={PloS ONE},
        volume={14},
        number={10},
        year={2019},
        doi={10.1371/journal.pone.0223792}
    }
    """
    )

    def _load_stimulus_events(
        self, timeline: dict[str, tp.Any]
    ) -> pd.DataFrame:
        """Load THINGS image events with absolute image file paths.

        Parameters
        ----------
        timeline:
            Timeline dict with ``subject``, ``session``, ``run``, ``task``.

        Returns
        -------
        pd.DataFrame
            Events table with ``type``, ``start``, ``duration``, ``filepath``,
            and ``category`` columns.
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
                "type": "Image",
                "start": float(row["onset"]),
                "duration": float(row["duration"]),
            }
            stim_file = row.get("stim_file")
            if stim_file and isinstance(stim_file, str):
                event["filepath"] = str(stimuli_dir / stim_file)
            category = row.get("trial_type")
            if category:
                event["category"] = str(category)
            rows.append(event)

        return pd.DataFrame(rows) if rows else pd.DataFrame()
