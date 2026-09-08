"""OOD movie-watching fMRI dataset.

Four subjects watched segments from a variety of movies and TV shows (Black and white
silent film "The Pawn Shop" (1916), stick figure animated short "World of Tomorrow" (2015),
and extracts from BBC Documentary series "Planet Earth" (2006, episode 2, "Mountains"),
from animated film "Princess Mononoke" (1997), from Hollywood movie "Pulp Fiction" (1994)
and from French-Canadian children's show "Passe-Partout" (1979; episodes #94 "Bon
Coup, Mauvais Coup" and #95 "Cause et Effet"), totalling ~2 hours of movie watching
(~20 minutes per movie/show) while undergoing 3T fMRI.

Each movie was split into two BOLD runs of 8-12 minutes each.

References
----------
* CNeuroMod documentation: https://docs.cneuromod.ca/latest/datasets/ood.html
* DataLad BIDS repo: https://github.com/courtois-neuromod/ood
* DataLad fMRIPrep repo: https://github.com/courtois-neuromod/ood.fmriprep
* DataLad timeseries repo: https://github.com/courtois-neuromod/ood.timeseries
* DataLad stimuli repo: https://github.com/courtois-neuromod/ood.stimuli
* DataLad transcripts repo: https://github.com/courtois-neuromod/ood.annotations
"""

from __future__ import annotations

import json
import typing as tp
from pathlib import Path

from neuralfetch_cneuromod.base import CNeuroModMovieStudy


class OOD(CNeuroModMovieStudy):
    """Courtois NeuroMod — *OOD* movie-watching fMRI dataset.

    Four subjects watched 2 hours of movie and TV show extracts while 
    undergoing 3T fMRI. Movie/TV show titles include *Pulp Fiction (1994)*,
    *Princess Mononoke (1997)*, *World of Tomorrow (2015)*, *The Pawn Shop (1916)*,
    *Passe-Partout (1979)*, and *Planet Earth (2006)*.

    Parameters
    ----------
    path:
        Root data directory.  Resolves ``{path}/ood/bids``,
        ``{path}/odd/fmriprep`` or ``{path}/ood/timeseries``,
        ``{path}/ood/stimuli`` and ``{path}/ood/annotations``.
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

    Example
    --------
    >>> study = OOD(path="path/to/cneuromod.all")
    >>> events = study.run()
    """

    TASK: tp.ClassVar[str] = "ood"
    BIDS_REPO: tp.ClassVar[str] = "ood"
    FMRIPREP_REPO: tp.ClassVar[str] = "ood.fmriprep"
    TIMESERIES_REPO: tp.ClassVar[str] = "ood.timeseries"
    STIMULI_REPO: tp.ClassVar[str] = "ood.stimuli"
    TRANSCRIPTS_REPO: tp.ClassVar[str] = "ood.annotations"
    MOVIES: list[str] = [
        "chaplin", "mononoke", "passepartout", "planetearth", "pulpfiction", "wot"]

    dataset_name: tp.ClassVar[str] = "CNeuroMod OOD"
    description: tp.ClassVar[str] = (
        "Four subjects watching 2 hours of movies / TV shows extracts during 3T fMRI."
    )
    bibtex: tp.ClassVar[str] = CNeuroModMovieStudy.bibtex


    # -----------------------------------------------------------------
    # Download pattern builders
    # -----------------------------------------------------------------

    def _stimuli_download_patterns(self) -> list[str]:
        """Build stimuli glob patterns for movie files (.mkv).

        Only movies segmented for individual runs are targeted:

        * e.g., ``task-chaplin1_video.mkv``

        Returns
        -------
        list[str]
            Glob patterns relative to the stimuli repository root, ready to
            be passed as ``datalad get`` arguments. Patterns are python glob
            compatible.
        """
        patterns = []
        for mvie in self.MOVIES:
            patterns.extend([
                # Movie stimuli MKVs shown for this movie-watching task
                f"{self._stimuli_dir}/{mvie}/task-{mvie}1_video.mkv",
                f"{self._stimuli_dir}/{mvie}/task-{mvie}2_video.mkv",
            ])
        return patterns

    def _annotations_download_patterns(self) -> list[str]:
        """Build annotation glob patterns for movie transcripts (.json).

        Transcripts for movies segmented for individual runs are targeted:

        * e.g., ``task-mononoke1_model-AA_transcript.json``

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
                f"{self._annotations_dir}/annotations/transcripts/"
                f"{mvie}/task-{mvie}*_model-AA_transcript.json",
            ])
        return patterns

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
            f"{self._stimuli_dir}/{seg_name[:-1]}"
            f"/task-{seg_name}_video.mkv",
        )
        if not mp.exists():
            raise FileNotFoundError(
                f"Movie file not found: {mp}\n"
                "Run study.download() or datalad get to fetch the content."
            )
        return mp


    # TODO: adjust from movie10 to OOD
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
            f"{seg_name[:-1]}/task-{seg_name}_model-AA_transcript.json",            
        )
        if not tp.exists():
            return {
                "transcript": "",
                "words": [],
            }
        with open(tp, "r") as file:
            transcript = json.load(file)

        return transcript

