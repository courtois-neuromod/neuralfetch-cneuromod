"""Triplets word task fMRI dataset.

Four subjects (sub-01, sub-02, sub-03 and sub-06) performed two word judgement tasks 
while undergoing 3T fMRI: a familiarity judgement for single words, and a similarity
(semantic odd-one-out) judgement between word triplets.

References
----------
* CNeuroMod documentation: https://docs.cneuromod.ca/datasets/triplets.html
* DataLad BIDS repo: https://github.com/courtois-neuromod/triplets
* DataLad fMRIPrep repo: https://github.com/courtois-neuromod/triplets.fmriprep
* DataLad timeseries repo: https://github.com/courtois-neuromod/triplets.timeseries
"""

from __future__ import annotations

import typing as tp
import numpy as np

from neuralfetch_cneuromod.base import CNeuroModStudy


class Triplets(CNeuroModStudy):
    """Courtois NeuroMod — triplets word judgement task dataset.

    Four subjects performed a word familiarity judgement task and an 
    odd-one-out judgement task on word triplets while undergoing 3T fMRI.

    3-run sessions were split evenly betweeen the two tasks and identified 
    with the task-wordsfamiliarity and task-triplets tags, respectively.

    Parameters
    ----------
    path:
        Root data directory.  Resolves ``{path}/triplets/bids`` and
        ``{path}/triplets/fmriprep`` or ``{path}/triplets/timeseries``,
    space:
        fMRIPrep output space (default ``"MNI152NLin2009cAsym"``).
    timeseries:
        Define to model pre-extracted, masked, denoised and normalized timeseries, 
        rather than the fMRIPrep BOLD derivatives. Select among ``"cneuromod2026"``, 
        ``"schaefer1000"`` (algonauts 2025 competition), ``"voxel_mni"`` or ``"voxel_native"``.
        (default ``None``) .
    subjects:
        Restrict data loading to a subset of subject labels
        (without ``sub-`` prefix).  ``None`` includes all available subjects.
    datalad_jobs:
        Parallel DataLad download jobs.

    Notes
    -----
    The bids events.tsv encodes each trial's timing, condition, word(s) 
    and subject response (button press).

    Examples
    --------
    >>> study = Triplets(path="path/to/cneuromod.all")
    >>> events = study.run()
    """

    TASK: tp.ClassVar[str] = "triplets"
    BIDS_REPO: tp.ClassVar[str] = "triplets"
    FMRIPREP_REPO: tp.ClassVar[str] = "triplets.fmriprep"
    TIMESERIES_REPO: tp.ClassVar[str] = "triplets.timeseries"

    dataset_name: tp.ClassVar[str] = "CNeuroMod Triplets"
    description: tp.ClassVar[str] = (
        "Four subjects performing a word(s) judgement task "
        "during 3T fMRI."
    )
    bibtex: tp.ClassVar[str] = CNeuroModStudy.bibtex


    # -----------------------------------------------------------------
    # Event loading
    # -----------------------------------------------------------------

    def _extract_stimulus_event(
        self,
        row: pd.Series,
    ) -> list[dict[str, tp.Any]]:
        """."""
        if "triplet_id" in row.index:
            """
            run of word triplet odd-one-out judgement task
            """
            events: list[dict[str, tp.Any]] =  [
                {
                    "type": "Word",
                    "text": row["w3"],
                    "start": float(row["onset"]),
                    "duration": float(row["duration"]),
                    "language": "en",
                    "modality": "read",
                    "extra": {
                        "position": "top",
                        "answer_choice": 3,
                        "repeat": row.values[8],
                        "triplet_id": row["triplet_id"],
                    },
                },
                {
                    "type": "Word",
                    "text": row["w2"],
                    "start": float(row["onset"]),
                    "duration": float(row["duration"]),
                    "language": "en",
                    "modality": "read",
                    "extra": {
                        "position": "middle",
                        "answer_choice": 2,
                        "repeat": row.values[8],
                        "triplet_id": row["triplet_id"],
                    },
                },
                {
                    "type": "Word",
                    "text": row["w1"],
                    "start": float(row["onset"]),
                    "duration": float(row["duration"]),
                    "language": "en",
                    "modality": "read",
                    "extra": {
                        "position": "bottom",
                        "answer_choice": 1,
                        "repeat": row.values[8],
                        "triplet_id": row["triplet_id"],
                    },
                },
            ]
            if not np.isnan(row["response_time"]):
                events.append(
                    {
                        "type": "Action",
                        "code": int(row["response_txt"]),   # TODO: check best practice? int: 1, 2, 3 answer choice
                        "start": float(row["answer_onset"]),
                        "extra": {
                            "word_choice": row[f"w{int(row['response_txt'])}"],
                            "answer_choice": int(row["response_txt"]),
                            "key": row["answer"],
                            "response_time": float(row["response_time"]),
                            "repeat": row.values[8],
                            "triplet_id": row["triplet_id"],
                        },
                    }
                )
        else:
            """
            run of single word familiarity judgement
            """
            events: list[dict[str, tp.Any]] =  [
                {
                    "type": "Word",
                    "text": row["word"],
                    "start": float(row["onset"]),
                    "duration": float(row["duration"]),
                    "language": "en",
                    "modality": "read",
                },
            ]
            if not np.isnan(row["response_time"]):
                familiarity_map = {
                    1: "unfamiliar", 2: "somewhat familiar", 3: "familiar",
                }       
                events.append(
                    {
                        "type": "Action",
                        "code": int(row["answer_text"]),   # TODO: check best practice? int: 1, 2, 3 answer choice
                        "start": float(row["answer_onset"]),
                        "extra": {
                            "familiarity": familiarity_map[int(row["answer_text"])],
                            "answer_choice": int(row["answer_text"]),
                            "key": row["answer"],
                            "response_time": float(row["response_time"]),
                        },
                    }
                )

        return events