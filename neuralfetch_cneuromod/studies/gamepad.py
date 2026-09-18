"""Gamepad motor task fMRI dataset.

Four subjects performed button-pressing motor task on an MRI-compatible videogame
controller (i.e., a gamepad) while undergoing 3T fMRI. Different fingers from both
hands were cued for short and long button-presses.

References
----------
* CNeuroMod documentation: https://docs.cneuromod.ca/latest/datasets/gamepad.html
* DataLad BIDS repo: https://github.com/courtois-neuromod/gamepad
* DataLad fMRIPrep repo: https://github.com/courtois-neuromod/gamepad.fmriprep
* DataLad timeseries repo: https://github.com/courtois-neuromod/gamepad.timeseries
"""

from __future__ import annotations

import typing as tp

from neuralfetch_cneuromod.base import CNeuroModStudy


class Gamepad(CNeuroModStudy):
    """Courtois NeuroMod — gamepad motor task dataset.

    Four subjects performed button-pressing tasks on a gamepad 
    using multiple fingers from both hands while undergoing 3T fMRI.

    Button-pressing conditions
    Duration : ['long', 'short']
    Key : ['a', 'b', 'd', 'l', 'r', 'u', 'x', 'y']
    Hand : ['l', 'r']

    Parameters
    ----------
    path:
        Root data directory.  Resolves ``{path}/gamepad/bids`` and
        ``{path}/gamepad/fmriprep`` or ``{path}/gamepad/timeseries``,
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
    The bids events.tsv encodes the timing and conditions of the key
    pressing events (which key, left or right hand, long or short presses).

    Examples
    --------
    >>> study = Gamepad(path="path/to/cneuromod.all")
    >>> events = study.run()
    """

    TASK: tp.ClassVar[str] = "gamepad"
    BIDS_REPO: tp.ClassVar[str] = "gamepad"
    FMRIPREP_REPO: tp.ClassVar[str] = "gamepad.fmriprep"
    TIMESERIES_REPO: tp.ClassVar[str] = "gamepad.timeseries"

    dataset_name: tp.ClassVar[str] = "CNeuroMod Gamepad"
    description: tp.ClassVar[str] = (
        "Four subjects performing a gamepad button-pressing task "
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
        events: dict[str, tp.Any] =  [
            {
                "type": "Stimulus",
                "start": float(row["onset"]),
                "duration": float(row["duration"]),
                "modality": "visual",
                "description": f"key-{row['key']}_hand-{row['lr_condition']}_len-{row['condition']}",
                "extra": {
                    "key": row["key"],
                    "length_condition": row["condition"],
                    "lr_condition": row["lr_condition"],
                },
            },
        ]
        if not np.isnan(row["key_press_time"]):
            # key press duration sometimes not logged for short button presses
            key_duration = 0.15 if np.isnan(row["key_duration"]) else float(row["key_duration"])
            events.append(
                {
                    "type": "Action",
                    "start": float(row["key_press_time"]),
                    "duration": key_duration,
                    "description": f"key-{row['key']}_hand-{row['lr_condition']}_len-{row['condition']}",
                    "extra": {
                        "key": row["key"],
                        "length_condition": row["condition"],
                        "lr_condition": row["lr_condition"],
                    },
                }
            )

        return events