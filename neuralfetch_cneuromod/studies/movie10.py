"""Movie10 movie-watching fMRI dataset.

Six subjects watched three Hollywood feature films ("The Bourne Supremacy" (2004), 
"The Wolf of Wall Street" (2013), "Hidden Figures" (2016)) and one BBC nature documentary 
("Life : Challenges of life, reptiles and amphibian mammals" (2009)) totalling ~10 hours of 
movie watching while undergoing 3T fMRI.

Hidden Figure and Life were both visioned twice to support reproducibility analyses.
Each movie was split into multiple BOLD runs of ~10-minute each.

References
----------
* CNeuroMod documentation: https://docs.cneuromod.ca/en/latest/DATASETS.html#movie10
* DataLad BIDS repo: https://github.com/courtois-neuromod/movie10
* DataLad fMRIPrep repo: https://github.com/courtois-neuromod/movie10.fmriprep
* DataLad timeseries repo: https://github.com/courtois-neuromod/movie10.timeseries
* DataLad stimuli repo: https://github.com/courtois-neuromod/movie10.stimuli
* DataLad transcripts repo: https://github.com/courtois-neuromod/movie10.annotations
"""

from __future__ import annotations

import json
import typing as tp

import pandas as pd

from neuralfetch_cneuromod import _utils
from neuralfetch_cneuromod.base import (
    CNeuroModStudy,
    _CNEUROMOD_GH,
)

class Movie10(CNeuroModStudy):
    """Courtois NeuroMod — *Movie10* movie-watching fMRI dataset.

    Six subjects watched 10 hours of Hollywood movies/BBC documentary while 
    undergoing 3T fMRI. Movie titles include *Life : Challenges of life, 
    reptiles and amphibian mammals (2009)* (shown twice), *Hidden Figures (2016)*
    (shown twice), *The Wolf of Wall Street (2013)*, and *The Bourne Supremacy (2004)*.

    Parameters
    ----------
    path:
        Root data directory.  Resolves ``{path}/movie10/bids``,
        ``{path}/movie10/fmriprep`` or ``{path}/movie10/timeseries``,
        ``{path}/movie10/stimuli`` and ``{path}/movie10/annotations``.
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

    Examples
    --------
    >>> study = Movie10(path="path/to/cneuromod.all")
    >>> events = study.run()
    """

    TASK: tp.ClassVar[str] = "movie10"
    BIDS_REPO: tp.ClassVar[str] = "movie10"
    FMRIPREP_REPO: tp.ClassVar[str] = "movie10.fmriprep"
    TIMESERIES_REPO: tp.ClassVar[str] = "movie10.timeseries"
    STIMULI_REPO: tp.ClassVar[str] = "movie10.stimuli"
    TRANSCRIPTS_REPO: tp.ClassVar[str] = "movie10.annotations"
    MOVIES: list[str] = ["bourne", "figures", "life", "wolf"]

    dataset_name: tp.ClassVar[str] = "CNeuroMod Movie10"
    description: tp.ClassVar[str] = (
        "Six subjects watching 10 hours of Hollywood movies / BBC documentary during 3T fMRI."
    )
    bibtex: tp.ClassVar[str] = CNeuroModStudy.bibtex


    # -----------------------------------------------------------------
    # Directory resolution
    # -----------------------------------------------------------------

    def model_post_init(self, log__: tp.Any) -> None:
        """Resolve stimuli and annotation subdirectory paths after init."""
        super().model_post_init(log__)
        self._stimuli_dir = self._resolve_subdir("stimuli", self._stimuli_repo_url())
        self._annotations_dir = self._resolve_subdir("annotations", self._annotations_repo_url())

    # -----------------------------------------------------------------
    # Directory accessors
    # -----------------------------------------------------------------

    @property
    def stimuli_dir(self) -> Path:
        """Path to the stimulus DataLad repository with 
        audio-visual movie files in .mkv format.

        Returns
        -------
        Path
            ``{path}/stimuli``
        """
        return self._stimuli_dir

    @property
    def annotations_dir(self) -> Path:
        """Path to the annotations DataLad repository with 
        time-stamped movie scripts (dialogue).

        Returns
        -------
        Path
            ``{path}/annotations``
        """
        return self._annotations_dir

    # -----------------------------------------------------------------
    # Repository URL / name helpers
    # -----------------------------------------------------------------

    def _stimuli_repo_name(self) -> str:
        """Repository name for the stimuli dataset (without ``.git`` suffix)."""
        return self.STIMULI_REPO

    def _annotations_repo_name(self) -> str:
        """Repository name for the annotations dataset."""
        return self.TRANSCRIPTS_REPO

    def _stimuli_repo_url(self) -> str:
        """GitHub SSH URL for the stimuli dataset."""
        return _CNEUROMOD_GH.format(repo=self.STIMULI_REPO)

    def _annotations_repo_url(self) -> str:
        """GitHub SSH URL for the annotations dataset."""
        return _CNEUROMOD_GH.format(repo=self.TRANSCRIPTS_REPO)

    # -----------------------------------------------------------------
    # Download pattern builders
    # -----------------------------------------------------------------

    def _stimuli_download_patterns(self) -> list[str]:
        """Build stimuli glob patterns for movie files (.mkv).

        Only movies segmented for individual runs are targeted:

        * e.g., ``bourne01.mkv``

        Returns
        -------
        list[str]
            Glob patterns relative to the movie repository root, ready to
            be passed as ``datalad get`` arguments. Patterns are python glob
            compatible.
        """
        patterns = []
        for mvie in self.MOVIES:
            patterns.extend([
                # Movie stimuli MKVs shown for this movie-watching task
                f"{self.path}/stimuli/{mvie}/{mvie}*.mkv",
            ])
        return patterns

    def _annotations_download_patterns(self) -> list[str]:
        """Build annotation glob patterns for movie transcripts (.json).

        Transcripts for movies segmented for individual runs are targeted:

        * e.g., ``bourne01.json``

        Returns
        -------
        list[str]
            Glob patterns relative to the annotation repository root, ready to
            be passed as ``datalad get`` arguments. Patterns are python glob
            compatible.
        """
        patterns = []
        for mvie in self.MOVIES:
            patterns.extend([
                # Movie dialogues transcribed with AssemblyAI speech-to-text
                f"{self.path}/annotations/transcripts/{mvie}/"
                f"movie10_{mvie}*_model-AA_transcript.json",
            ])
        return patterns

    # -----------------------------------------------------------------
    # Download
    # -----------------------------------------------------------------

    def _download(self) -> None:
        """Selectively fetch data files from stimuli and transcripts DataLad 
        repositories.

        Pulls files selectively by passing BIDS glob patterns to ``datalad get``.

        Pulled files include:
        * **stimuli** — cloned; only ``*.mkv`` matching individual runs 
        are fetched.  
        * **annotations** — cloned; only ``*.json`` matching individual runs 
        are fetched.  

        The patterns are built by :meth:`_stimuli_download_patterns` and
        :meth:`_annotations_download_patterns` from the current field values.
        """
        super()._download()

        stimuli_patterns = self._stimuli_download_patterns()
        self.logger.info(
            "[%s] stimuli patterns: %s", stimuli_patterns
        )
        _utils.datalad_get_list(stimuli_patterns, f"{self.path}/stimuli")

        transcript_patterns = self._annotations_download_patterns()
        self.logger.info(
            "[%s] transcript patterns (space=%s): %s",
            transcript_patterns,
        )
        _utils.datalad_get_list(transcript_patterns, f"{self.path}/annotations")

    # -----------------------------------------------------------------
    # Event loading
    # -----------------------------------------------------------------

    def _get_movie_path(self, timeline: dict[str, tp.Any]) -> Path:
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
            f"{self._stimuli_dir}/{seg_name[:-2]}"
            f"/{seg_name}.mkv",
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
            The transcript for the segmented movie shown during a given run.
        """
        if self.timeseries:
            seg_name = timeline['run'].split("_")[1][5:]
        else:
            seg_name = timeline['task']
        tp = Path(
            f"{self._annotations_dir}/annotations/transcripts/"
            f"{seg_name[:-2]}/movie10_{seg_name}_model-AA_transcript.json",            
        )
        if not tp.exists():
            return {
                "transcript": "",
                "words": [],
            }
        with open(tp, "r") as file:
            transcript = json.load(file)

        return transcript


    def _load_stimulus_events(
        self, timeline: dict[str, tp.Any], timeline_name: str,
    ) -> pd.DataFrame:
        """Load Movie10 stimulus events. Loads run-wise Video event with
        video clip file paths. Also extracts Word events from movie transcript.

        Detects FPS (as frequency) and duration from movie file.

        Parameters
        ----------
        timeline:
            Timeline dict with ``subject``, ``session``, ``run``, ``task``.
        timeline_name:
            Unique timeline identifier.

        Returns
        -------
        pd.DataFrame
            Table with Movie event (run-wise) and Word events from movie transcript.
        """
        movie_path = self._get_movie_path(timeline)
        movie_event: dict[str, tp.Any] = {
            "type": "Video",
            "start": 0.0,
            "filepath": movie_path,
            "timeline": timeline_name,
        }
        stimuli_events = [movie_event]

        transcript = self._load_transcript(timeline)
        for word in transcript["words"]:
            word_event : dict[str, tp.Any] = {
                "type": "Word",
                "text": word["word"],
                "start": word["start"],
                "stop": word["end"],
                "duration": word["end"] - word["start"],
                "language": "en",
                "modality": "heard",
                "timeline": timeline_name,
            }
            stimuli_events.append(word_event)
        if len(transcript["transcript"]):
            text_start = transcript["words"][0]["start"]
            text_stop = transcript["words"][-1]["end"]
            text_event : dict[str, tp.Any] = {
                "type": "Text",
                "text": transcript["transcript"],
                "start": text_start,
                "stop": text_stop,
                "duration": text_stop - text_start,
                "language": "en",
                "modality": "heard",
                "timeline": timeline_name,
            }
            stimuli_events.append(text_event)

        return pd.DataFrame(stimuli_events)
