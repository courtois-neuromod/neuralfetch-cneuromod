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
* CNeuroMod documentation: https://docs.cneuromod.ca/latest/datasets/petit-prince.html
* DataLad BIDS repo: https://github.com/courtois-neuromod/petit-prince
* DataLad fMRIPrep repo: https://github.com/courtois-neuromod/petit-prince.fmriprep
* DataLad timeseries repo: https://github.com/courtois-neuromod/petit-prince.timeseries
* DataLad stimuli repo: https://github.com/courtois-neuromod/petitprince.stimuli
* DataLad transcripts repo: https://github.com/courtois-neuromod/petit-prince.annotations
"""

from __future__ import annotations

import json
import typing as tp
from pathlib import Path

import pandas as pd

from neuralfetch_cneuromod.base import CNeuroModAudioStudy


class PetitPrince(CNeuroModAudioStudy):
    """Courtois NeuroMod — *Petit Prince* audiobook listening dataset (EN and FR).

    Five subjects listened to an audiobook verion of Le Petit Prince (1943) in French and 
    then in English during 3T fMRI.  Each BOLD run covers one segment of the book.

    Parameters
    ----------
    path:
        Root data directory.  Resolves ``{path}/petit-prince/bids``,
        ``{path}/petit-prince/fmriprep`` or ``{path}/petit-prince/timeseries``,
        ``{path}/petit-prince/stimuli`` and ``{path}/petit-prince/annotations``.
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
    >>> study = PetitPrince(path="/data/to/cneuromod.all")
    >>> events = study.run()
    """

    TASK: tp.ClassVar[str] = "petit-prince"
    BIDS_REPO: tp.ClassVar[str] = "petit-prince"
    FMRIPREP_REPO: tp.ClassVar[str] = "petit-prince.fmriprep"
    TIMESERIES_REPO: tp.ClassVar[str] = "petit-prince.timeseries"
    STIMULI_REPO: tp.ClassVar[str] = "petit-prince.stimuli"
    TRANSCRIPTS_REPO: tp.ClassVar[str] = "petit-prince.annotations"
    LANGUAGES: list[str] = ["EN", "FR"]

    dataset_name: tp.ClassVar[str] = "CNeuroMod Le Petit Prince"
    description: tp.ClassVar[str] = (
        "Five subjects listening to Le Petit Prince (1943) audiobook "
        "in English and French during 3T fMRI."
    )
    bibtex: tp.ClassVar[str] = CNeuroModAudioStudy.bibtex

    # -----------------------------------------------------------------
    # Download pattern builders
    # -----------------------------------------------------------------

    def _stimuli_download_patterns(self) -> list[str]:
        """Build stimuli glob patterns for audio files (.wav).

        Audio files are downloaded from their source OpenNeuro dset
        repository installed as a submodule. Only French (FR) and
        English (EN) audiobooks are targeted:

        * e.g., ``task-lppEN_section-1.wav``

        Returns
        -------
        list[str]
            Glob patterns relative to the main repository root, ready to
            be passed as ``datalad get`` arguments. Patterns are python glob
            compatible.
        """
        patterns = []
        for lang in self.LANGUAGES:
            patterns.extend([
                f"{self.path}/stimuli/OpenNeuroDatasets/ds003643/"
                f"stimuli/task-lpp{lang}_section*.wav",
            ])
        return patterns


    def _annotations_download_patterns(self) -> list[str]:
        """Build annotation glob patterns for audio transcripts (.json).

        Transcripts for audiobook segmented into individual runs are
        targeted in English and French:

        * e.g., ``movie10_bourne01_model-AA_transcript.json``

        Returns
        -------
        list[str]
            Glob patterns relative to the annotation repository root, ready to
            be passed as ``datalad get`` arguments. Patterns are python glob
            compatible.
        """
        patterns = []
        for lang in self.LANGUAGES:
          patterns.extend([
                # Audiobook transcribed with AssemblyAI speech-to-text
                f"{self.path}/annotations/annotations/transcripts/{lang}/"
                f"task-lpp{lang}_section-*_model-AA_transcript.json",
            ])
        return patterns

    # -----------------------------------------------------------------
    # Event loading
    # -----------------------------------------------------------------

    def _get_stimulus_path(self):
        """
        Return the full path of the audiobook segment file (.wav) presented
        during a given run ('timeline').

        Parameters
        ----------
        timeline:
            Timeline dictionary with keys ``subject``, ``session``, ``run``,
            ``task``.

        Returns
        -------
        Path
            The path to the audiobook segment file presented during a given run.

        Raises
        ------
        FileNotFoundError
            If the .wav file does not exist (DataLad content not fetched).
        """
        if self.timeseries:
            _, task, run, _ = timeline['run'].split("_")
            lang = task[-2:]
        else:
            task = f"task-{timeline['task']}"
            run = timeline['run']
            lang = task[-2:]
        if lang == "EN":
            seg_name = f"{task}_section-{run[-1]}"
        else:
            seg_name = f"{task}_section_{run[-1]}"

        ap = Path(
            f"{self._stimuli_dir}/OpenNeuroDatasets/ds003643/"
            f"stimuli/{seg_name}.wav",
        )
        if not ap.exists():
            raise FileNotFoundError(
                f"Audio file not found: {ap}\n"
                "Run study.download() or datalad get to fetch the content."
            )
        return ap


    def _load_transcript(self):
        """
        Load the speech-to-text transcript of the audiobook segment
        presented during a given run ('timeline').

        Parameters
        ----------
        timeline:
            Timeline dictionary with keys ``subject``, ``session``, ``run``,
            ``task``.

        Returns
        -------
        dict
            The transcript for the audiobook segment presented during a given run.
        str
            The language of the transcript ("en" or "fr")
        """
        if self.timeseries:
            _, task, run, _ = timeline['run'].split("_")
            seg_name = f"{task}_section-{run[-1]}"
            lang = task[-2:]
        else:
            seg_name = f"task-{timeline['task']}_section-{timeline['run']}"
            lang = timeline['task'][-2:]
        tp = Path(
            f"{self._annotations_dir}/annotations/transcripts/"
            f"{lang}/{seg_name}_model-AA_transcript.json",            
        )
        if not tp.exists():
            return {
                "transcript": "",
                "words": [],
            }, lang.lower()

        with open(tp, "r") as file:
            transcript = json.load(file)

        return transcript, lang.lower()


    def _load_stimulus_events(
        self, timeline: dict[str, tp.Any], event_root: str,
    ) -> pd.DataFrame:
        """Load audio stimulus events. Loads run-wise Audio event with
        audio file paths. Also extracts Word events from audio transcript.

        Duration and frequency are auto-detected from audio file if not provided.

        Parameters
        ----------
        timeline:
            Timeline dict with ``subject``, ``session``, ``run``, ``task``.
        event_root:
            Unique event file identifier.

        Returns
        -------
        pd.DataFrame
            Table with Audio event (run-wise) and Word events from speech2text
            audio transcript.
        """
        audio_path = self._get_stimulus_path(timeline)
        audio_event: dict[str, tp.Any] = {
            "type": "Audio",
            "start": 0.0,
            "filepath": audio_path,
        }
        stimuli_events = [audio_event]

        transcript, lang = self._load_transcript(timeline)
        for word in transcript["words"]:
            word_event : dict[str, tp.Any] = {
                "type": "Word",
                "text": word["word"],
                "start": word["start"],
                "stop": word["end"],
                "duration": word["end"] - word["start"],
                "language": lang,
                "modality": "heard",
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
                "language": lang,
                "modality": "heard",
            }
            stimuli_events.append(text_event)

        return pd.DataFrame(stimuli_events)
