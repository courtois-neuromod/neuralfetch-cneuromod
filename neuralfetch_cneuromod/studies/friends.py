"""Friends TV show fMRI dataset.

Six subjects watched episodes from the Friends sitcom while while undergoing 3T fMRI. 
One subject (``sub-04``) watched seasons 1-4, while the remaining five subjects watched 
seasons 1-6 (totalling >50 hours of audio-visual stimulation).

Most episodes are split into two BOLD runs (labelled a and b), and double episodes are split
into up to four BOLD runs (a, b, c and d).  Stimuli are provided as video clip files (.mkv)
in the ``stimuli`` repository. Corresponding movie transcripts are provided in the annotations
repository.

References
----------
* CNeuroMod documentation: https://docs.cneuromod.ca/latest/datasets/friends.html
* DataLad BIDS repo: https://github.com/courtois-neuromod/friends
* DataLad fMRIPrep repo: https://github.com/courtois-neuromod/friends.fmriprep
* DataLad timeseries repo: https://github.com/courtois-neuromod/friends.timeseries
* DataLad stimuli repo: https://github.com/courtois-neuromod/friends.stimuli
* DataLad annotations repo: https://github.com/courtois-neuromod/friends.annotations
"""

from __future__ import annotations

import json
import typing as tp
from pathlib import Path

from neuralfetch_cneuromod.base import CNeuroModMovieStudy


class Friends(CNeuroModMovieStudy):
    """Courtois NeuroMod — *Friends* TV-watching fMRI dataset.

    Six subjects (01, 02, 03, 04, 05, 06) watched several seasons of the 
    Friends sitcom across multiple scanning sessions. One subject (04) watched 
    seasons 1-4, while the remaining five subjects watched seasons 1-6. 
    Each Friends episode was split into ~12-minute segments (two for regular 
    episodes, four for double episodes) corresponding to BOLD runs.
    
    Parameters
    ----------
    path:
        Root data directory.  Resolves ``{path}/friends/bids``,
        ``{path}/friends/fmriprep`` or ``{path}/friends/timeseries``,
        ``{path}/friends/stimuli`` and ``{path}/friends/annotations``.
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
    * Stimuli (mkv video clips) are stored in the ``stimuli/`` repository.
    * Movie transcripts are stored in the ``annotations/`` repository.

    Examples
    --------
    >>> study = Friends(path="path/to/cneuromod.all")
    >>> events = study.run()
    """

    TASK: tp.ClassVar[str] = "friends"
    BIDS_REPO: tp.ClassVar[str] = "friends"
    FMRIPREP_REPO: tp.ClassVar[str] = "friends.fmriprep"
    TIMESERIES_REPO: tp.ClassVar[str] = "friends.timeseries"
    STIMULI_REPO: tp.ClassVar[str] = "friends.stimuli"
    TRANSCRIPTS_REPO: tp.ClassVar[str] = "friends.annotations"

    dataset_name: tp.ClassVar[str] = "CNeuroMod Friends"
    description: tp.ClassVar[str] = (
        "Six subjects watching Friends TV show seasons 1-6 (≈50 h) "
        "during 3T fMRI acquisition."
    )
    bibtex: tp.ClassVar[str] = CNeuroModMovieStudy.bibtex


    # -----------------------------------------------------------------
    # Download pattern builders
    # -----------------------------------------------------------------

    def _stimuli_download_patterns(self) -> list[str]:
        """Build stimuli glob patterns for movie files (.mkv).

        Only episodes segmented for individual runs are targeted:

        * e.g., ``friends_s01e01a.mkv``

        Returns
        -------
        list[str]
            Glob patterns relative to the stimuli repository root, ready to
            be passed as ``datalad get`` arguments. Patterns are python glob
            compatible.
        """
        return [
            f"{self._stimuli_dir}/s*/friends_s0*e*[abcd].mkv",
        ]

    def _annotations_download_patterns(self) -> list[str]:
        """Build annotation glob patterns for episode transcripts (.json).

        Transcripts for movies segmented for individual runs are targeted:

        * e.g., ``friends_s01e01a_model-AA_desc-wUtter_transcript.json``

        Returns
        -------
        list[str]
            Glob patterns relative to the annotation repository root, ready to
            be passed as ``datalad get`` arguments. Patterns are python glob
            compatible.
        """
        return [
            f"{self._annotations_dir}/automated_transcription/s*/"
            "friends_s0*e*_model-AA_desc-wUtter_transcript.json",  # TODO: update desc-wSpeaker
        ]

    # -----------------------------------------------------------------
    # Event loading
    # -----------------------------------------------------------------

    def _get_stimulus_path(self, timeline: dict[str, tp.Any]) -> Path:
        """
        Return the full path of the segmented movie file (.mkv) shown during a 
        given run ('timeline').

        Parameters
        ----------
        timeline:
            Timeline dictionary with keys ``subject``, ``session``, ``run``,
            ``task``.

        Returns
        -------
        Path
            The path to the segmented movie file shown during a given run.

        Raises
        ------
        FileNotFoundError
            If the .mkv file does not exist (DataLad content not fetched).
        """
        if self.timeseries:
            seg_name = timeline['run'].split("_")[1][5:]
        else:
            seg_name = timeline['task']
        mp = Path(
            f"{self._stimuli_dir}/s{seg_name[2]}"
            f"/friends_{seg_name}.mkv",
        )
        if not mp.exists():
            raise FileNotFoundError(
                f"Movie file not found: {mp}\n"
                "Run study.download() or datalad get to fetch the content."
            )
        return mp


    def _load_transcript(self, timeline: dict[str, tp.Any]) -> dict:
        """
        Load the speech-to-text transcript of the segmented movie 
        shown during a given run ('timeline').

        Parameters
        ----------
        timeline:
            Timeline dictionary with keys ``subject``, ``session``, ``run``,
            ``task``.

        Returns
        -------
        dict
            The transcript for the segmented episode shown during a given run.
        """
        if self.timeseries:
            seg_name = timeline['run'].split("_")[1][5:]
        else:
            seg_name = timeline['task']
        tp = Path(  # TODO: adjust based on friends.annotations structure
            f"{self._annotations_dir}/annotations/automated_transcription/"
            f"s{seg_name[2]}/friends_{seg_name}_model-AA_desc-wUtter_transcript.json",            
        )
        if not tp.exists():
            return {
                "transcript": "",
                "words": [],
            }
        with open(tp, "r") as file:
            transcript = json.load(file)

        return transcript