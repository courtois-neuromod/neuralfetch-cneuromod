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

import typing as tp

import pandas as pd

from neuralfetch_cneuromod.base import CNeuroModStudy
#from neuralfetch_cneuromod._utils import events_path, load_events_tsv


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
        for mvie in MOVIES:
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
        for mvie in MOVIES:
            patterns.extend([
                # Movie dialogues transcribed with AssemblyAI speech-to-text
                f"{self.path}/annotations/{mvie}/movie10_{mvie}*_model-AA_transcript.json",
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
        logger.info(
            "[%s] stimuli patterns: %s", cls_name, stimuli_patterns
        )
        _utils.datalad_get_list(stimuli_patterns, f"{self.path}/stimuli")

        transcript_patterns = self._annotations_download_patterns()
        logger.info(
            "[%s] transcript patterns (space=%s): %s",
            cls_name, transcript_patterns,
        )
        _utils.datalad_get_list(transcript_patterns, f"{self.path}/annotations")

    # -----------------------------------------------------------------
    # Event loading
    # -----------------------------------------------------------------

    def _get_movie_dur(self, timeline: dict[str, tp.Any]) -> tuple[Path, int]:
        """"""
        # TODO: implement get movie freq, duration, file path
        pass


    def _load_stimulus_events(
        self, timeline: dict[str, tp.Any], timeline_name: str,
    ) -> pd.DataFrame:
        """Load Movie10 stimulus events with video clip file paths.

        Parameters
        ----------
        timeline:
            Timeline dict with ``subject``, ``session``, ``run``, ``task``.
        timeline_name:
            Unique timeline identifier.

        Returns
        -------
        pd.DataFrame
            Events table with ``type``, ``start``, ``duration``, ``filepath``.
        """
        # TODO: fix below to process movies and transcript words as events
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
