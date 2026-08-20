"""Base study class shared by all Courtois NeuroMod dataset fetchers.

:class:`CNeuroModStudy` extends :class:`neuralset.events.study.Study` and
centralises:

* Directory layout resolution for the submodule structure
  (e.g., ``{study}/bids`` and ``{study}/fmriprep`` or ``{study}/timeseries``).
* DataLad download logic via selective ``datalad get`` with success-file idempotency.
* BOLD loading via :func:`~neuralfetch_cneuromod._utils.load_bold_masked`.
* Construction of a ``neuralset``-compatible events :class:`pandas.DataFrame`
  that exposes ``Fmri``, ``Stimulus``, and ``Events`` event types.

Concrete study classes only need to:

1. Set the :attr:`TASK` class variable and class-level variables.
2. Define :meth:`_extract_stimulus_event` to generate stimulus events from 
   `events.tsv` files in the raw BIDS dataset. Override 
   :meth:`_load_stimulus_events` to attach stimulus
   information at the whole-run level.
3. Optionally override :meth:`iter_timelines` when the default subject /
   session / run discovery logic is insufficient.
"""

from __future__ import annotations

import logging
import typing as tp
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import pydantic

from datalad import api as dl

from exca.steps.backends import Cached
from neuralset.events import study as _study

from . import _utils
_study._set_dir_permissions = _utils._set_dir_permissions


# GitHub URL for cneuromod.all repository, under which  
# CNeuroMod datalad repositories are nested as a collection
# of submodules organized per study
_CNEUROMOD_ALL = "https://github.com/courtois-neuromod/cneuromod.all.git"

# GitHub base URL for CNeuroMod datalad repositories
_CNEUROMOD_GH = "https://github.com/courtois-neuromod/{repo}.git"


class CNeuroModStudy(_study.Study):
    """Abstract base class for all Courtois NeuroMod study fetchers.

    Subclasses must set the :attr:`TASK` class variable and other class 
    variabltes. 
    
    Subclasses may override :meth:`iter_timelines`, as well as
    :meth:`_load_stimulus_events` and / or :meth:`_extract_stimulus_event.

    Parameters
    ----------
    path:
        Root directory under which CNeuroMod data lives.  Expected layout::

            path/cneuromod.all        ← pre-installed parent repository
            ├── {StudyName}/          ← auto-resolved subfolder
            │   ├── bids/             ← raw BIDS dataset (DataLad repo)
            │   ├── fmriprep/         ← fMRIPrep derivatives (DataLad repo)
            │   └── timeseries/       ← masked and denoised BOLD timeseries (DataLad repo)

        Alternatively, *path* can point directly to the study-level directory
        (i.e. the one that contains ``bids/`` and ``fmriprep/`` sub-dirs).
    space:
        fMRIPrep output space template (default ``"MNI152NLin2009cAsym"``).
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

    Class Variables
    ---------------
    TASK : str
        BIDS task label (e.g., "movie10").  **Must** be set by every concrete subclass.
    BIDS_REPO : str
        Name of the raw BIDS GitHub repository under the ``courtois-neuromod``
        organisation.  Defaults to the lower-cased class name.
    FMRIPREP_REPO : str
        Name of the fMRIPrep derivative GitHub repository.
        Defaults to ``"{BIDS_REPO}.fmriprep"``.
    """

    # -----------------------------------------------------------------
    # Class-level configuration — override in concrete subclasses
    # -----------------------------------------------------------------

    #: BIDS task label (must be overridden).
    TASK: tp.ClassVar[str] = ""

    #: GitHub repository name for the raw BIDS data.  Defaults to the
    #: lower-cased class name (e.g. ``"friends"`` for the ``Friends`` class).
    BIDS_REPO: tp.ClassVar[str] = ""

    #: GitHub repository name for the fMRIPrep and timeseries 
    # derivative datasets, respectivaly.
    FMRIPREP_REPO: tp.ClassVar[str] = ""
    TIMESERIES_REPO: tp.ClassVar[str] = ""

    # To be replaced with CNeuroMod data paper citation in BibTeX format.
    bibtex: tp.ClassVar[str] = """
    @article{boyle2020CCNposter,
        title={CNeuroMod, an open fMRI dataset with diverse naturalistic & 
        controlled tasks to build NeuroAI models},
        author={Boyle, Julie and Pinsard, Basile and St-Laurent, Marie 
        and Bellec, Lune},
        howpublished = {Poster presented at the OHBM 2026 Annual Meeting},
        address      = {Bordeaux, France},
        year={2026},
        month = {July},
    }
    """

    licence: tp.ClassVar[str] = (
        "CC0 (subjects 01, 02, 03, 05, 06) / Registered access — "
        "see https://www.cneuromod.ca/"
    )

    url: tp.ClassVar[str] = "https://www.cneuromod.ca/"

    logger: ClassVar[logging.Logger] = logging.getLogger(__name__)

    # -----------------------------------------------------------------
    # Pydantic fields
    # -----------------------------------------------------------------

    #: fMRIPrep output space template.
    space: str | None = _utils.DEFAULT_SPACE
    #: Define to model pre-extracted timeseries rather than fMRIprep output
    timeseries: str | None = None
    #: Restrict to a subset of subjects (labels without ``sub-`` prefix).
    subjects: list[str] | None = None
    #: Parallel jobs for ``datalad get`` during download.
    datalad_jobs: int = 1

    # Private attributes set during model_post_init
    _bids_dir: Path = pydantic.PrivateAttr(default=None)  # type: ignore[assignment]
    _fmriprep_dir: Path = pydantic.PrivateAttr(default=None)  # type: ignore[assignment]
    _timeseries_dir: Path = pydantic.PrivateAttr(default=None)  # type: ignore[assignment]

    # -----------------------------------------------------------------
    # Pydantic lifecycle
    # -----------------------------------------------------------------

    def model_post_init(self, log__: tp.Any) -> None:
        """Resolve bids/ and fmriprep/ subdirectory paths after init."""
        super().model_post_init(log__)
        # self.path has already been resolved to the study subfolder by the
        # parent Study.model_post_init (which appends the class name if needed).
        self._bids_dir = self._resolve_subdir("bids", self._bids_repo_url())
        if self.timeseries is None:
            self._fmriprep_dir = self._resolve_subdir("fmriprep", self._fmriprep_repo_url())
        else:
            self._timeseries_dir = self._resolve_subdir("timeseries", self._timeseries_repo_url())
        self.timelines.infra = Cached(folder=self.infra.folder)

    # -----------------------------------------------------------------
    # Directory resolution
    # -----------------------------------------------------------------

    def _resolve_subdir(self, folder: str, repo_url: str) -> Path:
        """Return full the path to a study's cloned sub-repository.

        If the sub-repository exists at ``self.path / folder`` (pre-cloned 
        manually or automatically during a previous study instantiation), 
        the script returns the repository's full path. 

        If the sub-repository does not exist, the script uses the repository's
        url to clone it with DataLad as ``self.path / folder``.   

        Parameters
        ----------
        folder:
            Sub-directory name under :attr:`path` (``"bids"``, 
            ``"fmriprep" or "timeseries"``).
        repo_url:
            Repository GitHub URL.

        Returns
        -------
        Path
            The full path to the cloned sub-repository.
        """
        sub_path = self.path / folder

        if not sub_path.exists():
            dl.clone(source=repo_url, path=sub_path)

        return sub_path

    # -----------------------------------------------------------------
    # Directory accessors
    # -----------------------------------------------------------------

    @property
    def bids_dir(self) -> Path:
        """Path to the raw BIDS DataLad repository.

        Returns
        -------
        Path
            ``{path}/bids``
        """
        return self._bids_dir

    @property
    def fmriprep_dir(self) -> Path:
        """Path to the fMRIPrep DataLad repository.

        Returns
        -------
        Path
            ``{path}/fmriprep``
        """
        return self._fmriprep_dir

    @property
    def timeseries_dir(self) -> Path:
        """Path to the timeseries DataLad repository.

        Returns
        -------
        Path
            ``{path}/timeseries``
        """
        return self._timeseries_dir

    # -----------------------------------------------------------------
    # Repository URL / name helpers
    # -----------------------------------------------------------------

    def _bids_repo_name(self) -> str:
        """Repository name for the raw BIDS dataset (without ``.git`` suffix)."""
        return self.BIDS_REPO

    def _fmriprep_repo_name(self) -> str:
        """Repository name for the fMRIPrep derivative dataset."""
        return self.FMRIPREP_REPO

    def _timeseries_repo_name(self) -> str:
        """Repository name for the timeseries derivative dataset."""
        return self.TIMESERIES_REPO

    def _bids_repo_url(self) -> str:
        """GitHub SSH URL for the raw BIDS dataset."""
        repo = self.BIDS_REPO or self.__class__.__name__.lower()
        return _CNEUROMOD_GH.format(repo=repo)

    def _fmriprep_repo_url(self) -> str:
        """GitHub SSH URL for the fMRIPrep derivative dataset."""
        return _CNEUROMOD_GH.format(repo=self.FMRIPREP_REPO)

    def _timeseries_repo_url(self) -> str:
        """GitHub SSH URL for the timeseries derivative dataset."""
        return _CNEUROMOD_GH.format(repo=self.TIMESERIES_REPO)

    # -----------------------------------------------------------------
    # Download pattern builders
    # -----------------------------------------------------------------

    def _subject_globs(self) -> list[str]:
        """Return a list of ``sub-`` glob prefixes for the configured subjects.

        When :attr:`subjects` is a non-empty list, returns a list of specific
        prefixes (e.g., ``["sub-01", "sub-03"]``). When :attr:`subjects` is
        ``None``, returns a single wildcard ``["sub-*"]``.

        Returns
        -------
        list[str]
            A list of glob prefixes.
        """
        if self.subjects:
            return [f"sub-{s}" for s in self.subjects]
        return ["sub-*"]

    def _bids_download_patterns(self) -> list[str]:
        """Build BIDS glob patterns for task design files only.

        Only files needed for study design / event modelling are targeted:

        * ``*_events.tsv`` — trial onset / duration / type annotations

        Returns
        -------
        list[str]
            Glob patterns relative to the BIDS repository root, ready to be
            passed as ``datalad get`` arguments. Patterns are python glob
            compatible.
        """
        patterns = []
        for sub in self._subject_globs():
            patterns.extend([
                # Functional events TSVs for this task
                f"{self._bids_dir}/{sub}/ses-*/func/{sub}_ses-*_task-*_events.tsv",
            ])
        return patterns

    def _fmriprep_download_patterns(self) -> list[str]:
        """Build fMRIPrep derivative glob patterns for the configured space.

        Only files required for analysis in the configured output space are
        targeted:

        * ``desc-preproc_bold.nii.gz`` — preprocessed BOLD in ``space``
        * ``desc-confounds_timeseries.tsv`` — nuisance confound regressors
        * ``desc-brain_mask.nii.gz`` — brain mask in ``space``

        All patterns are restricted to the subjects given by the :attr:`subjects` 
        (or all subjects when ``subjects`` is ``None``).

        Returns
        -------
        list[str]
            Glob patterns relative to the fMRIPrep repository root. Patterns
            are python glob compatible.
        """
        space = f"space-{self.space}"
        patterns = []
        for sub in self._subject_globs():
            patterns.extend([
                # Preprocessed BOLD in the target space
                f"{self._fmriprep_dir}/{sub}/ses-*/func/{sub}_ses-*_"
                f"task-*_{space}_desc-preproc_bold.nii.gz",
                
                # Confound regressors
                f"{self._fmriprep_dir}/{sub}/ses-*/func/{sub}_ses-*_"
                f"task-*_desc-confounds_timeseries.tsv",
                
                # Brain mask in the target space
                f"{self._fmriprep_dir}/{sub}/ses-*/func/{sub}_ses-*_"
                f"task-*_{space}_desc-brain_mask.nii.gz",
            ])
        return patterns

    def _timeseries_download_patterns(self) -> list[str]:
        """Build timeseries derivative glob patterns for the configured space.

        Only files required for analysis in the configured output space are
        targeted:

        * ``desc-preproc_bold.nii.gz`` — preprocessed BOLD in ``space``

        All patterns are restricted to the timeseries specified by :attr:`timeseries`, 
        and to the subjects given by :attr:`subjects` (or all subjects when 
        ``subjects`` is ``None``).

        Returns
        -------
        list[str]
            Glob patterns relative to the timeseries repository root. Patterns
            are python glob compatible.
        """
        task = self.TASK
        tseries = self.timeseries
        ts_desc = _utils.TSERIES_DESCRIPT[tseries]
        space = f"space-{self.space}"
        patterns = []
        for sub in self._subject_globs():
            patterns.extend([
                # BOLD timeseries in the target space
                f"{self._timeseries_dir}/timeseries/{tseries}/{sub}/{sub}_task-"
                f"{task}_{space}_{ts_desc}_timeseries.h5",
            ])
        return patterns

    # -----------------------------------------------------------------
    # Download
    # -----------------------------------------------------------------

    def _download(self) -> None:
        """Selectively fetch data files from BIDS and fMRIPrep / 
        timeseries DataLad repositories.

        Pulls files selectively by passing BIDS glob patterns to ``datalad get``.

        Pulled files include:
        * **BIDS** — cloned; only ``*_events.tsv`` and ``*_scans.tsv``
          matching the configured subjects are fetched.  Stimuli are
          left as git-annex pointers.
        * **fMRIPrep** — cloned; only the preprocessed BOLD, confounds TSV,
          and brain mask for the configured subjects are fetched.
        * **timeseries** — cloned; only the .hdf5 files, with timeseries nested per
        session and run, are fecthed for the configured ``timeseries`` and subjects 
        are fetched.

        The patterns are built by :meth:`_bids_download_patterns`,
        :meth:`_fmriprep_download_patterns` and :meth:`_timeseries_download_patterns` 
        from the current field values.
        """
        cls_name = self.__class__.__name__

        bids_patterns = self._bids_download_patterns()
        self.logger.info(
            "[%s] BIDS patterns: %s", cls_name, bids_patterns
        )
        _utils.datalad_get_list(bids_patterns, f"{self._bids_dir}")

        if self.timeseries is None:
            fmriprep_patterns = self._fmriprep_download_patterns()
            self.logger.info(
                "[%s] fMRIPrep patterns (space=%s): %s",
                cls_name, self.space, fmriprep_patterns,
            )
            _utils.datalad_get_list(fmriprep_patterns, f"{self._fmriprep_dir}")

        else:
            timeseries_patterns = self._timeseries_download_patterns()
            self.logger.info(
                "[%s] timeseries patterns (timeseries=%s): %s",
                cls_name, self.timeseries, timeseries_patterns,
            )
            _utils.datalad_get_list(timeseries_patterns, f"{self._timeseries_dir}")

    # -----------------------------------------------------------------
    # Timeline iteration
    # -----------------------------------------------------------------

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Iterate over all available (subject, session, run) triples.

        * **BOLD** — Uses :func:`~neuralfetch_cneuromod._utils.iter_bids_runs` to discover
        available preprocessed BOLD files in the fMRIPrep directory.
        * **Timeseries** — Uses :func:`~neuralfetch_cneuromod._utils.iter_tseries_runs` 
        to discover pre-extracted timeseries nested in .h5 files in the timeseries directory.

        Yields
        ------
        dict
            Keys: ``subject`` (str), ``session`` (str), ``file_path`` (str), 
            ``task`` (str | None), ``run`` (str | None).

        Raises
        ------
        FileNotFoundError
            If :attr:`fmriprep_dir` does not exist.
        """
        if self.timeseries is None:
            yield from _utils.iter_bids_runs(
                self._fmriprep_dir,
                subjects=self.subjects,
                space=self.space,
            )
        else:
            yield from _utils.iter_tseries_runs(
                self._timeseries_dir,
                task=self.TASK,
                subjects=self.subjects,
                timeseries=self.timeseries,
                space=self.space,
            )

    # -----------------------------------------------------------------
    # Event loading
    # -----------------------------------------------------------------

    def _get_scan_dur(self, timeline: dict[str, tp.Any]) -> tuple[str, int]:
        """Load a preprocessed BOLD run as a :class:`nibabel.Nifti1Image`, 
        return its full path and its number of volumes (duration in TRs).

        Parameters
        ----------
        timeline:
            Timeline dictionary with keys ``subject``, ``session``, ``run``,
            ``task``.

        Returns
        -------
        str
            The path to the fMRIPrep preprocessed BOLD file.
        int
            The number of volumes (TRs) in the preprocessed BOLD image.

        Raises
        ------
        FileNotFoundError
            If the BOLD file does not exist (DataLad content not fetched).
        """
        bp = timeline['file_path']
        if not Path(bp).exists():
            raise FileNotFoundError(
                f"BOLD file not found: {bp}\n"
                "Run study.download() or datalad get to fetch the content."
            )
        return bp, nib.load(str(bp)).shape[-1]


    def _get_tseries_dur(self, timeline: dict[str, tp.Any]) -> tuple[str, int]:
        """Open an hdf5 file of nested pre-extracted timeseries (one file per subject), 
        return its full path and the number of time points (duration in TRs) for a 
        given run ('timeline').

        Parameters
        ----------
        timeline:
            Timeline dictionary with keys ``subject``, ``session``, ``run``,
            ``task``.

        Returns
        -------
        str
            The path to the subject's .hdf5 file that contains the run's timeseries.
        int
            The number of time points (TRs) in the run's timeseries.

        Raises
        ------
        FileNotFoundError
            If the HDF5 file does not exist (DataLad content not fetched).
        """
        tp = timeline['file_path']
        if not Path(tp).exists():
            raise FileNotFoundError(
                f"HDF5 file not found: {tp}\n"
                "Run study.download() or datalad get to fetch the content."
            )

        with h5py.File(tp, "r") as f:
            n_TRs = np.array(f[f"{timeline['session']}"][f"{timeline['run']}"]).shape[0]

        return tp, n_TRs


    def _load_stimulus_events(
        self, timeline: dict[str, tp.Any], event_root: str,
    ) -> pd.DataFrame:
        """Load stimulus/behavioural events for *timeline*.

        Default implementation reads the BIDS ``*_events.tsv`` file when the
        raw BIDS directory exists, and returns an empty DataFrame otherwise.

        Subclasses should override to attach stimulus metadata (e.g. video file
        paths, image identifiers).

        Parameters
        ----------
        timeline:
            Timeline dictionary with keys ``subject``, ``session``, ``run``,
            ``task``.
        event_root:
            Unique event file identifier.

        Returns
        -------
        pd.DataFrame
            Events DataFrame with at least ``type``, ``start``, ``duration``
            columns (neuralset format).  May be empty.
        """
        if not self._bids_dir.exists():
            return pd.DataFrame()

        event_wc = (
            f"{self._bids_dir}/sub-{timeline['subject']}"
            f"/*{timeline['session']}/func/{event_root}*events.tsv"
        )
        ep_list = sorted(glob.glob(event_wc))
        if len(ep_list) != 1:
            self.logger.debug("No unique events file found: %s", event_wc)
            return pd.DataFrame()

        bids_events = pd.read_csv(ep_list[0], sep="\t")
        # Map BIDS columns to neuralset conventions
        rows = []
        for _, row in bids_events.iterrows():
            event = self._extract_stimulus_event(row)
            if event:
                rows.append(event)
        return pd.DataFrame(rows)


    def _extract_stimulus_event(
        self,
        row: pd.Series,
    ) -> dict[str, tp.Any] | None:
        """Implement file processing logic in subclasses.
        
        The returned ``event`` dict must at least contain ``type``, ``start``,
        and ``duration``.

        e.g.,
        event: dict[str, tp.Any] = {
            "type": str(row.get("trial_type", "Stimulus")),
            "start": float(row["onset"]),
            "duration": float(row["duration"]),
        }
        # Carry over any additional columns (e.g. stim_file, response_time)
        for idx in row.index:
            if idx not in ("onset", "duration", "trial_type"):
                event[idx] = row[idx]

        Returns
        -------
        Any
            The loaded data
        """
        return None


    def _cls_kwargs(self) -> dict[str, tp.Any]:
        """Descriptor for the study instance parametrization"""
        cls_kwargs: tp.Any = self.model_dump(serialize_as_any=True, exclude_defaults=True)
        # Exclude standard fields from class kwargs
        for p in ["infra", "timelines", "path", "name", "query", "timeseries", "space"]:
            cls_kwargs.pop(p, None)
        if cls_kwargs:
            # should the class parameter be part of the timeline? or does
            # it select a subset? the behavior is unclear and should be
            # specified precisely first.
            msg = "Class parameters are not yet supported, bring up your use-case!"
            raise RuntimeError(msg)

        return cls_kwargs


    def _load_timeline_events(
        self, timeline: dict[str, tp.Any]
    ) -> pd.DataFrame:
        """Assemble the full events DataFrame for a single timeline.

        Builds two kinds of rows:

        * **BOLD** — one row pointing to the preprocessed BOLD file via a
          :class:`neuralset.events.study.SpecialLoader`.
        * **Stimulus / trial events** — rows from the BIDS events TSV if
          the file exists. Some tasks (friends, movie10) or sub-tasks (some language
          localizers) have no events files.

        Parameters
        ----------
        timeline:
            Timeline dictionary with at least ``subject``, ``session``,  
            ``file_path`` , ``task`` and/or ``run`` keys.

        Returns
        -------
        pd.DataFrame
            Combined events table in neuralset format.
        """
        # --- fMRI event row ---
        tr_s = _utils.DEFAULT_TR
        if self.timeseries is None:
            bold_path, n_TRs = self._get_scan_dur(timeline)
            fmri_row: dict[str, tp.Any] = {
                "type": "Fmri",
                "start": 0.0,
                "duration": float(n_TRs) * tr_s,
                "frequency": 1.0 / tr_s,
                "filepath": bold_path,
                "mask_filepath": bold_path.replace("preproc_bold.", "brain_mask."),
                "confounds_filepath": bold_path.replace(
                    f"_space-{self.space}_desc-preproc_bold.nii.gz",
                    "_desc-confounds_timeseries.tsv"),
                "space": self.space,
                "preproc": "fmriprep",
            }
            event_root = Path(bold_path).name.split(
                "_space")[0].replace("_part-mag", "")
        else:
            tseries_path, n_TRs = self._get_tseries_dur(timeline)
            fmri_row: dict[str, tp.Any] = {
                "type": "Timeseries",
                "start": 0.0,
                "duration": float(n_TRs) * tr_s,
                "frequency": 1.0 / tr_s,
                "filepath": tseries_path,
                "timeseries": self.timeseries,
                "space": self.space,
            }
            event_root = (
                f"sub-{timeline['subject']}_"
                f"{timeline['run'].split('_timeseries')[0]}")

        # --- Stimulus / behavioural events ---
        stim_events = self._load_stimulus_events(timeline, event_root)

        all_rows = [pd.DataFrame([fmri_row])]
        if not stim_events.empty:
            all_rows.append(stim_events)

        return pd.concat(all_rows, ignore_index=True)


class CNeuroModAudioStudy(CNeuroModStudy):
    """Abstract base class for all Courtois NeuroMod movie-watching and 
    audio-listening study fetchers, including Petit Prince, Narratives,
    Friends, Movie10 and OOD.

    Subclasses must set the :attr:`TASK` class variable and must override
    :meth:`_get_stimulus_path`, :meth:`_load_transcript`, :meth:`_stimuli_download_patterns`,
    :meth:`_annotations_download_patterns` and :meth:`_load_stimulus_events`.

    Expected layout::

        path/cneuromod.all        ← pre-installed parent repository
        ├── {StudyName}/          ← auto-resolved subfolder
        │   ├── bids/             ← raw BIDS dataset (DataLad repo)
        │   ├── fmriprep/         ← fMRIPrep derivatives (DataLad repo)
        │   ├── timeseries/       ← masked and denoised BOLD timeseries (DataLad repo)
        │   ├── stimuli/          ← Movie/audio files (.mkv/.wav) presented to participants
        │   └── annotations/      ← Movie annotations, including transcripts

    Class Variables
    ---------------
    STIMULI_REPO : str
        Name of the stimuli GitHub repository under the ``courtois-neuromod``
        organisation. Defaults to ``"{BIDS_REPO}.stimuli"`.
    TRANSCRIPTS_REPO : str
        Name of the stimulus annotations GitHub repository.
        Defaults to ``"{BIDS_REPO}.annotations"``.
    """

    # -----------------------------------------------------------------
    # Class-level configuration — override in concrete subclasses
    # -----------------------------------------------------------------

    #: GitHub repository name for the stimuli data (movie or audio).
    STIMULI_REPO: tp.ClassVar[str] = ""
    #: GitHub repository name for the stimulus annotaiton data 
    # (movie or audio transcript).
    TRANSCRIPTS_REPO: tp.ClassVar[str] = ""

    # Private attributes set during model_post_init
    _stimuli_dir: Path = pydantic.PrivateAttr(default=None)  # type: ignore[assignment]
    _annotations_dir: Path = pydantic.PrivateAttr(default=None)  # type: ignore[assignment]

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
        audio-visual files in .mkv, .wav or .mp3 format.

        Returns
        -------
        Path
            ``{path}/stimuli``
        """
        return self._stimuli_dir

    @property
    def annotations_dir(self) -> Path:
        """Path to the annotations DataLad repository with 
        time-stamped audio / movie transcripts (narration or dialogue).

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
        """Build stimuli glob patterns for audio (.wav) or movie (.mkv)
        files.

        Must override in concrete subclasses

        Returns
        -------
        list[str]
            Glob patterns relative to the movie repository root, ready to
            be passed as ``datalad get`` arguments. Patterns are python glob
            compatible.
        """
        return []

    def _annotations_download_patterns(self) -> list[str]:
        """Build annotation glob patterns for audio / movie transcripts (.json).

        Must override in concrete subclasses

        Returns
        -------
        list[str]
            Glob patterns relative to the annotation repository root, ready to
            be passed as ``datalad get`` arguments. Patterns are python glob
            compatible.
        """
        return []

    # -----------------------------------------------------------------
    # Download
    # -----------------------------------------------------------------

    def _download(self) -> None:
        """Selectively fetch data files from stimuli and transcripts DataLad 
        repositories.

        Pulls files selectively by passing BIDS glob patterns to ``datalad get``.

        Pulled files include:
        * **stimuli** — cloned; only ``*.mkv`` or ``*.wav`` matching individual
        runs are fetched.  
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
        _utils.datalad_get_list(stimuli_patterns, f"{self._stimuli_dir}")

        transcript_patterns = self._annotations_download_patterns()
        self.logger.info(
            "[%s] transcript patterns (space=%s): %s",
            transcript_patterns,
        )
        _utils.datalad_get_list(transcript_patterns, f"{self._annotations_dir}")

    # -----------------------------------------------------------------
    # Event loading
    # -----------------------------------------------------------------

    def _load_transcript(self, timeline: dict[str, tp.Any]) -> dict:
        """
        Load the speech-to-text transcript of the audio / movie
        presented during a given run ('timeline').

        Must override in concrete subclasses

        Parameters
        ----------
        timeline:
            Timeline dictionary with keys ``subject``, ``session``, ``run``,
            ``task``.

        Returns
        -------
        dict
            The transcript for the audio / movie presented during a given run.
        """
        return {
            "transcript": "",
            "words": [],
        }

    def _get_stimulus_path(self, timeline: dict[str, tp.Any]) -> Path:
        """
        Return the full path of the segmented audio (.wav) or movie (.mkv) file
        presented during a given run ('timeline').

        Must override in concrete subclasses

        Parameters
        ----------
        timeline:
            Timeline dictionary with keys ``subject``, ``session``, ``run``,
            ``task``.

        Returns
        -------
        Path
            The path to the segmented movie / audio file presented during a given run.
        """
        return Path(".")


    def _load_stimulus_events(
        self, timeline: dict[str, tp.Any], event_root: str,
    ) -> pd.DataFrame:
        """Load movie/audio stimulus events and corresponding transcripts.
        
        Must override in concrete subclasses

        Parameters
        ----------
        timeline:
            Timeline dict with ``subject``, ``session``, ``run``, ``task``.
        event_root:
            Unique event file identifier.

        Returns
        -------
        pd.DataFrame
            Table with Audio / Movie event (run-wise) and Word events from
            audio / movie transcript.
        """
        return pd.DataFrame()


class CNeuroModMovieStudy(CNeuroModAudioStudy):
    """Abstract base class for all Courtois NeuroMod movie-watching study
    fetchers, including Friends, Movie10 and OOD.

    Subclasses must set the :attr:`TASK` class variable and must override
    :meth:`_get_stimulus_path`, :meth:`_load_transcript`, :meth:`_stimuli_download_patterns`
    and :meth:`_annotations_download_patterns`.

    Expected layout::

        path/cneuromod.all        ← pre-installed parent repository
        ├── {StudyName}/          ← auto-resolved subfolder
        │   ├── bids/             ← raw BIDS dataset (DataLad repo)
        │   ├── fmriprep/         ← fMRIPrep derivatives (DataLad repo)
        │   ├── timeseries/       ← masked and denoised BOLD timeseries (DataLad repo)
        │   ├── stimuli/          ← Movie files (.mkv) shown to participants
        │   └── annotations/      ← Movie annotations, including transcripts

    Class Variables
    ---------------
    STIMULI_REPO : str
        Name of the stimuli GitHub repository under the ``courtois-neuromod``
        organisation. Defaults to ``"{BIDS_REPO}.stimuli"`.
    TRANSCRIPTS_REPO : str
        Name of the stimulus annotations GitHub repository.
        Defaults to ``"{BIDS_REPO}.annotations"``.
    """

    # -----------------------------------------------------------------
    # Event loading
    # -----------------------------------------------------------------

    def _load_stimulus_events(
        self, timeline: dict[str, tp.Any], event_root: str,
    ) -> pd.DataFrame:
        """Load movie stimulus events. Loads run-wise Video event with
        video clip file paths. Also extracts Word events from movie transcript.

        Detects FPS (as frequency) and duration from movie file.

        Parameters
        ----------
        timeline:
            Timeline dict with ``subject``, ``session``, ``run``, ``task``.
        event_root:
            Unique event file identifier.

        Returns
        -------
        pd.DataFrame
            Table with Movie event (run-wise) and Word events from speech2text
            movie transcript.
        """
        movie_path = self._get_stimulus_path(timeline)
        movie_event: dict[str, tp.Any] = {
            "type": "Video",
            "start": 0.0,  # 0s onset for Friends, Movie10 and OOD
            "filepath": movie_path,
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
            }
            stimuli_events.append(text_event)

        return pd.DataFrame(stimuli_events)


class CNeuroModVideoGameStudy(CNeuroModStudy):
    """Abstract base class for all Courtois NeuroMod videogame-playing study
    fetchers, including Shinobi, Mario, MarioStars and Mario3.

    Subclasses must set the :attr:`TASK` class variable and may need to override
    :meth:`_get_replay_events` and :meth:`_replays_download_patterns`.

    Expected layout::

        path/cneuromod.all        ← pre-installed parent repository
        ├── {StudyName}/          ← auto-resolved subfolder
        │   ├── bids/             ← raw BIDS dataset (DataLad repo)
        │   ├── fmriprep/         ← fMRIPrep derivatives (DataLad repo)
        │   └── timeseries/       ← masked and denoised BOLD timeseries (DataLad repo)

    Note
    -----
    Game data (stored as .bk2) and replays (.mp4) are stored in the bids repository.
    """

    # -----------------------------------------------------------------
    # Download pattern builders
    # -----------------------------------------------------------------

    def _replays_download_patterns(self) -> list[str]:
        """Build glob patterns for game replay files (.mp4).

        Targets multiple game replays per run. Each replay includes up
        to three attempts ("lives") to clear a game level.

        Returns
        -------
        list[str]
            Glob patterns relative to the BIDS repository root, ready to be
            passed as ``datalad get`` arguments. Patterns are python glob
            compatible.
        """
        patterns = []
        for sub in self._subject_globs():
            patterns.extend([
                # Game replays (.mp4) for this videogaming task
                f"{self._bids_dir}/{sub}/ses-*/gamelogs/{sub}_ses-*_task-*_recording.mp4",
            ])
        return patterns

    # -----------------------------------------------------------------
    # Download
    # -----------------------------------------------------------------

    def _download(self) -> None:
        """Selectively fetch replay files from bids DataLad repository.

        Pulls files selectively by passing BIDS glob patterns to ``datalad get``.

        Pulled files include:
        * **replays** — cloned; only ``*.mp4`` matching multiple game replays
        per run are fetched.  

        The patterns are built by :meth:`_replays_download_patterns` from the 
        current field values.
        """
        super()._download()

        replay_patterns = self._replays_download_patterns()
        self.logger.info(
            "[%s] replay patterns: %s", replay_patterns
        )
        _utils.datalad_get_list(replay_patterns, f"{self._bids_dir}")


    # -----------------------------------------------------------------
    # Event loading
    # -----------------------------------------------------------------

    def _get_replay_events(
        self, timeline: dict[str, tp.Any], events_path: str,
    ) -> list[dict[str, tp.Any]]:
        """
        Return a list of dict with info for every level played 
        during a given run ('timeline'). Info includes the level,
        repetition number, onset (relative to run/timeline onset)
        and game replay file path (.mp4).

        Parameters
        ----------
        timeline:
            Timeline dictionary with keys ``subject``, ``session``, ``run``,
            ``task``.
        events_path:
            Path to run's *events.tsv file, which includes game play levels, 
            repetition, onset and duration.

        Returns
        -------
        Path
            List of dict with replay file paths and info (onset, level, rep num)
            for every level played during a given run.
        """
        df = pd.read_csv(events_path, sep="\t")
        game_events = df[df["trial_type"]=="gym-retro_game"]
        replay_path = Path(events_path.replace("/func", "/gamelogs")).parent

        r_list = [{
            "type": "Video",
            "start": row.onset,
            "filepath": replay_path / f"{Path(row.stim_file).name.replace(
                '.bk2', '_recording.mp4')}",
            "level": row.level,
            "attempt": row.stim_file.split("rep-")[-1].split(".")[0],
        } for i, row in enumerate(game_events.itertuples())]
        
        return [d for d in r_list if d['filepath'].exists()]


    def _load_stimulus_events(
        self, timeline: dict[str, tp.Any], event_root: str,
    ) -> pd.DataFrame:
        """Load game replay events. Loads run-wise replay events with their
        full file paths.

        Detects FPS (as frequency) and duration from .mp4 file.

        Parameters
        ----------
        timeline:
            Timeline dict with ``subject``, ``session``, ``run``, ``task``.
        event_root:
            Unique event file identifier.

        Returns
        -------
        pd.DataFrame
            Table with replay events for a given run.
        """
        if not self._bids_dir.exists():
            return pd.DataFrame()

        # run num: fmriprep -> bids for mario, mario3, mariostars and shinobi
        event_wc = (
            f"{self._bids_dir}/sub-{timeline['subject']}/*{timeline['session']}"
            f"/func/{event_root.replace('run-', 'run-0')}*events.tsv"
        )
        ep_list = [x for x in sorted(glob.glob(event_wc)) if not 'desc' in x]
        if len(ep_list) != 1:
            self.logger.debug("No unique events file found: %s", event_wc)
            return pd.DataFrame()

        replay_events = self._get_replay_events(timeline, ep_list[0])

        return pd.DataFrame(replay_events)