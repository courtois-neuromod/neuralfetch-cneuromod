"""Base study class shared by all Courtois NeuroMod dataset fetchers.

:class:`CNeuroModStudy` extends :class:`neuralset.events.study.Study` and
centralises:

* Directory layout resolution for the two-submodule structure
  (``{study}/bids`` and ``{study}/fmriprep``).
* DataLad download logic via :class:`neuralfetch.download.Datalad` (clone +
  selective ``datalad get`` with success-file idempotency).
* BOLD loading via :func:`~neuralfetch_cneuromod._utils.load_bold_masked`.
* Construction of a ``neuralset``-compatible events :class:`pandas.DataFrame`
  that exposes ``Fmri``, ``Stimulus``, and ``Events`` event types.

Concrete study classes only need to:

1. Set the :attr:`TASK` class variable.
2. Optionally override :meth:`_load_stimulus_events` to attach stimulus
   information from the raw BIDS dataset.
3. Optionally override :meth:`iter_timelines` when the default subject /
   session / run discovery logic is insufficient.
"""

from __future__ import annotations

import logging
import typing as tp
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pydantic

from neuralfetch.download import Datalad

from neuralset.events import study as _study

from . import _utils


logger = logging.getLogger(__name__)

# GitHub base URL for CNeuroMod datalad repositories
_CNEUROMOD_GH = "https://github.com/courtois-neuromod/{repo}.git"


class CNeuroModStudy(_study.Study):
    """Abstract base class for all Courtois NeuroMod study fetchers.

    Subclasses must set the :attr:`TASK` class variable and may override
    :meth:`_load_stimulus_events` and :meth:`iter_timelines`.

    Parameters
    ----------
    path:
        Root directory under which CNeuroMod data lives.  Expected layout::

            path/
            ├── {StudyName}/          ← auto-resolved subfolder
            │   ├── bids/             ← raw BIDS dataset (DataLad repo)
            │   └── fmriprep/         ← fMRIPrep derivatives (DataLad repo)

        Alternatively, *path* can point directly to the study-level directory
        (i.e. the one that contains ``bids/`` and ``fmriprep/`` sub-dirs).
    space:
        fMRIPrep output space template (default ``"MNI152NLin2009cAsym"``).
    resolution:
        Template resolution label (default ``"2"``).
    subjects:
        Restrict data loading to a subset of subject labels
        (without ``sub-`` prefix).  ``None`` includes all available subjects.
    datalad_jobs:
        Number of parallel jobs used by ``datalad get`` during download.

    Class Variables
    ---------------
    TASK : str
        BIDS task label.  **Must** be set by every concrete subclass.
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

    #: GitHub repository name for the fMRIPrep derivative dataset.
    FMRIPREP_REPO: tp.ClassVar[str] = ""

    #: CNeuroMod data paper citation in BibTeX format.
    bibtex: tp.ClassVar[str] = """
    @article{boyle2023iterative,
        title={An iterative approach for fitting generative models in an
        individual fMRI study using BOLD data collected across many imaging
        sessions (the Courtois NeuroMod Project)},
        author={Boyle, Jéremy and Pinsard, Basile and Bellec, Pierre and
        others},
        journal={NeuroImage},
        year={2023},
        doi={10.1016/j.neuroimage.2023.120137}
    }
    """

    licence: tp.ClassVar[str] = (
        "CC0 (subjects 01, 03, 05) / Registered access — "
        "see https://www.cneuromod.ca/"
    )

    url: tp.ClassVar[str] = "https://www.cneuromod.ca/"

    # -----------------------------------------------------------------
    # Pydantic fields
    # -----------------------------------------------------------------

    #: fMRIPrep output space template.
    space: str | None = _utils.DEFAULT_SPACE
    #: Template resolution label.
    resolution: str | None = _utils.DEFAULT_RESOLUTION
    #: Restrict to a subset of subjects (labels without ``sub-`` prefix).
    subjects: list[str] | None = None
    #: Parallel jobs for ``datalad get`` during download.
    datalad_jobs: int = 1

    # Private attributes set during model_post_init
    _bids_dir: Path = pydantic.PrivateAttr(default=None)  # type: ignore[assignment]
    _fmriprep_dir: Path = pydantic.PrivateAttr(default=None)  # type: ignore[assignment]

    # -----------------------------------------------------------------
    # Pydantic lifecycle
    # -----------------------------------------------------------------

    def model_post_init(self, log__: tp.Any) -> None:
        """Resolve bids/ and fmriprep/ subdirectory paths after init."""
        super().model_post_init(log__)
        # self.path has already been resolved to the study subfolder by the
        # parent Study.model_post_init (which appends the class name if needed).
        self._bids_dir = self._resolve_subdir("bids", self._bids_repo_name())
        self._fmriprep_dir = self._resolve_subdir("fmriprep", self._fmriprep_repo_name())
        self.infra_timelines.cluster = None

    # -----------------------------------------------------------------
    # Directory resolution
    # -----------------------------------------------------------------

    def _resolve_subdir(self, folder: str, repo_name: str) -> Path:
        """Return the path to a DataLad-managed or manually cloned sub-repo.

        :class:`neuralfetch.download.Datalad` clones into
        ``dset_dir / folder / repo_name``.  Pre-existing manually cloned repos
        are often placed directly at ``path / folder``.  This method tries both
        conventions in order.

        Parameters
        ----------
        folder:
            Sub-directory name under :attr:`path` (``"bids"`` or
            ``"fmriprep"``).
        repo_name:
            Repository name as derived from the GitHub URL (used by the
            :class:`~neuralfetch.download.Datalad` convention).

        Returns
        -------
        Path
            The resolved path (which may not yet exist on disk).
        """
        # Datalad-managed layout: path/folder/repo_name
        datalad_managed = self.path / folder / repo_name
        if datalad_managed.exists():
            return datalad_managed
        # Flat layout (manually cloned): path/folder
        return self.path / folder

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

    def _check_dirs(self) -> None:
        """Raise informative errors when expected directories are absent."""
        if not self._fmriprep_dir.exists():
            raise FileNotFoundError(
                f"fMRIPrep directory not found: {self._fmriprep_dir}\n"
                f"Run study.download() or clone the DataLad repo manually:\n"
                f"  datalad clone {self._fmriprep_repo_url()} {self._fmriprep_dir}"
            )

    # -----------------------------------------------------------------
    # Repository URL / name helpers
    # -----------------------------------------------------------------

    def _bids_repo_name(self) -> str:
        """Repository name for the raw BIDS dataset (without ``.git`` suffix)."""
        url = self._bids_repo_url()
        name = Path(url).name
        return name[:-4] if name.endswith(".git") else name

    def _fmriprep_repo_name(self) -> str:
        """Repository name for the fMRIPrep derivative dataset."""
        url = self._fmriprep_repo_url()
        name = Path(url).name
        return name[:-4] if name.endswith(".git") else name

    def _bids_repo_url(self) -> str:
        """GitHub SSH URL for the raw BIDS dataset."""
        repo = self.BIDS_REPO or self.__class__.__name__.lower()
        return _CNEUROMOD_GH.format(repo=repo)

    def _fmriprep_repo_url(self) -> str:
        """GitHub SSH URL for the fMRIPrep derivative dataset."""
        if self.FMRIPREP_REPO:
            return _CNEUROMOD_GH.format(repo=self.FMRIPREP_REPO)
        bids_repo = self.BIDS_REPO or self.__class__.__name__.lower()
        return _CNEUROMOD_GH.format(repo=f"{bids_repo}.fmriprep")

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
        * ``*_scans.tsv`` — session-level scan metadata (when present)

        Stimuli (video, audio, image files) are intentionally excluded because
        they can be very large and are not required for fMRI modelling.

        Returns
        -------
        list[str]
            Glob patterns relative to the BIDS repository root, ready to be
            passed as ``datalad get`` arguments. Patterns are python glob
            compatible.
        """
        task = self.TASK
        patterns = []
        for sub in self._subject_globs():
            patterns.extend([
                # Functional events TSVs for this task (with session)
                f"{sub}/ses-*/func/{sub}_ses-*_task-{task}_*_events.tsv",
                # Functional events TSVs for this task (without session)
                f"{sub}/func/{sub}_task-{task}_*_events.tsv",
            ])
        return patterns

    def _fmriprep_download_patterns(self) -> list[str]:
        """Build fMRIPrep derivative glob patterns for the configured space.

        Only files required for analysis in the configured output space are
        targeted:

        * ``desc-preproc_bold.nii.gz`` — preprocessed BOLD in ``space``/``res``
        * ``desc-confounds_timeseries.tsv`` — nuisance confound regressors
        * ``desc-brain_mask.nii.gz`` — brain mask in ``space``/``res``

        All patterns are restricted to the task specified by :attr:`TASK` and
        to the subjects given by :attr:`subjects` (or all subjects when
        ``subjects`` is ``None``).

        Returns
        -------
        list[str]
            Glob patterns relative to the fMRIPrep repository root. Patterns
            are python glob compatible.
        """
        task = self.TASK
        space = self.space
        res = self.resolution
        space_res = f"space-{space}"
        if res is not None:
            space_res += f"_res-{res}"
        patterns = []
        for sub in self._subject_globs():
            patterns.extend([
                # Preprocessed BOLD in the target space / resolution (with session)
                f"{sub}/ses-*/func/{sub}_ses-*_task-{task}*_{space_res}_desc-preproc_bold.nii.gz",
                # Preprocessed BOLD (without session)
                f"{sub}/func/{sub}_task-{task}*_{space_res}_desc-preproc_bold.nii.gz",
                
                # Confound regressors (with session)
                f"{sub}/ses-*/func/{sub}_ses-*_task-{task}*_desc-confounds_timeseries.tsv",
                # Confound regressors (without session)
                f"{sub}/func/{sub}_task-{task}*_desc-confounds_timeseries.tsv",
                
                # Brain mask in the target space / resolution (with session)
                f"{sub}/ses-*/func/{sub}_ses-*_task-{task}*_{space_res}_desc-brain_mask.nii.gz",
                # Brain mask (without session)
                f"{sub}/func/{sub}_task-{task}*_{space_res}_desc-brain_mask.nii.gz",
            ])
        return patterns

    # -----------------------------------------------------------------
    # Download
    # -----------------------------------------------------------------

    def _download(self) -> None:
        """Clone and selectively fetch BIDS and fMRIPrep DataLad repositories.

        Delegates to :class:`_CNeuroModDatalad` (a subclass of
        :class:`~neuralfetch.download.Datalad`) which passes BIDS glob
        patterns directly to ``datalad get``, enabling file-level selectivity:

        * **BIDS** — cloned; only ``*_events.tsv`` and ``*_scans.tsv``
          matching the configured task and subjects are fetched.  Stimuli are
          left as git-annex pointers.
        * **fMRIPrep** — cloned; only the preprocessed BOLD, confounds TSV,
          and brain mask for the configured ``space`` / ``resolution`` and
          subjects are fetched.

        The patterns are built by :meth:`_bids_download_patterns` and
        :meth:`_fmriprep_download_patterns` from the current field values.
        Idempotency is managed by the success files written by
        :class:`~neuralfetch.download.Datalad`.

        After download, :attr:`bids_dir` and :attr:`fmriprep_dir` are
        re-resolved to point at the correct paths.

        Raises
        ------
        RuntimeError
            If DataLad / git-annex is not installed or operations fail.
        """
        cls_name = self.__class__.__name__
        bids_patterns = self._bids_download_patterns()
        fmriprep_patterns = self._fmriprep_download_patterns()

        logger.info(
            "[%s] BIDS patterns: %s", cls_name, bids_patterns
        )
        logger.info(
            "[%s] fMRIPrep patterns (space=%s, res=%s): %s",
            cls_name, self.space, self.resolution, fmriprep_patterns,
        )

        # --- Raw BIDS: clone + fetch events/scans only ---
        bids_dl = Datalad(
            study=f"{cls_name}_bids",
            dset_dir=self.path,
            folder="bids",
            repo_url=self._bids_repo_url(),
            threads=self.datalad_jobs,
            paths=bids_patterns,
        )
        bids_dl.download()

        # --- fMRIPrep: clone + fetch space/subject-specific files ---
        fmriprep_dl = Datalad(
            study=f"{cls_name}_fmriprep",
            dset_dir=self.path,
            folder="fmriprep",
            repo_url=self._fmriprep_repo_url(),
            threads=self.datalad_jobs,
            paths=fmriprep_patterns,
        )
        fmriprep_dl.download()

        # Re-resolve directory pointers now that repos exist on disk
        self._bids_dir = self._resolve_subdir("bids", bids_dl.repo_name)
        self._fmriprep_dir = self._resolve_subdir("fmriprep", fmriprep_dl.repo_name)

    # -----------------------------------------------------------------
    # Timeline iteration
    # -----------------------------------------------------------------

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Iterate over all available (subject, session, run) triples.

        Uses :func:`~neuralfetch_cneuromod._utils.iter_bids_runs` to discover
        available preprocessed BOLD files in the fMRIPrep directory.

        Yields
        ------
        dict
            Keys: ``subject`` (str), ``session`` (str | None),
            ``run`` (str | None), ``task`` (str).

        Raises
        ------
        FileNotFoundError
            If :attr:`fmriprep_dir` does not exist.
        """
        self._check_dirs()
        yield from _utils.iter_bids_runs(
            self._fmriprep_dir,
            self._bids_dir if self._bids_dir.exists() else None,
            self.TASK,
            subjects=self.subjects,
            space=self.space,
            resolution=self.resolution,
        )

    # -----------------------------------------------------------------
    # Event loading
    # -----------------------------------------------------------------

    def _load_raw(self, timeline: dict[str, tp.Any]) -> nib.Nifti1Image:
        """Load a preprocessed BOLD run as a :class:`nibabel.Nifti1Image`.

        Parameters
        ----------
        timeline:
            Timeline dictionary with keys ``subject``, ``session``, ``run``,
            ``task``.

        Returns
        -------
        nibabel.Nifti1Image
            The preprocessed BOLD image.

        Raises
        ------
        FileNotFoundError
            If the BOLD file does not exist (DataLad content not fetched).
        """
        sub = timeline["subject"]
        ses = timeline.get("session")
        run = timeline.get("run")
        task = timeline.get("task", self.TASK)

        bp = _utils.bold_path(
            self._fmriprep_dir, sub, task,
            session=ses, run=run,
            space=self.space, resolution=self.resolution,
        )
        if not bp.exists():
            raise FileNotFoundError(
                f"BOLD file not found: {bp}\n"
                "Run study.download() or datalad get to fetch the content."
            )
        return nib.load(str(bp))  # type: ignore[return-value]

    def _load_stimulus_events(
        self, timeline: dict[str, tp.Any]
    ) -> pd.DataFrame:
        """Load stimulus/behavioural events for *timeline*.

        Default implementation reads the BIDS ``*_events.tsv`` file when the
        raw BIDS directory exists, and returns an empty DataFrame otherwise.

        Subclasses should override this method to attach dataset-specific
        stimulus metadata (e.g. video file paths, image identifiers).

        Parameters
        ----------
        timeline:
            Timeline dictionary with keys ``subject``, ``session``, ``run``,
            ``task``.

        Returns
        -------
        pd.DataFrame
            Events DataFrame with at least ``type``, ``start``, ``duration``
            columns (neuralset format).  May be empty.
        """
        if not self._bids_dir.exists():
            return pd.DataFrame()

        sub = timeline["subject"]
        ses = timeline.get("session")
        run = timeline.get("run")
        task = timeline.get("task", self.TASK)

        ep = _utils.events_path(self._bids_dir, sub, task, session=ses, run=run)
        if not ep.exists():
            logger.debug("No events file found: %s", ep)
            return pd.DataFrame()

        bids_events = _utils.load_events_tsv(ep)
        # Map BIDS columns to neuralset conventions
        rows = []
        for _, row in bids_events.iterrows():
            event: dict[str, tp.Any] = {
                "type": str(row.get("trial_type", "Stimulus")),
                "start": float(row["onset"]),
                "duration": float(row["duration"]),
            }
            # Carry over any additional columns (e.g. stim_file, response_time)
            for col in bids_events.columns:
                if col not in ("onset", "duration", "trial_type"):
                    event[col] = row[col]
            rows.append(event)
        return pd.DataFrame(rows)

    def _load_timeline_events(
        self, timeline: dict[str, tp.Any]
    ) -> pd.DataFrame:
        """Assemble the full events DataFrame for a single timeline.

        Builds two kinds of rows:

        * **Fmri** — one row pointing to the preprocessed BOLD file via a
          :class:`neuralset.events.study.SpecialLoader`.
        * **Stimulus / trial events** — rows from the BIDS events TSV (if
          the raw BIDS repo is present).

        Parameters
        ----------
        timeline:
            Timeline dictionary with at least ``subject``, ``task`` keys.

        Returns
        -------
        pd.DataFrame
            Combined events table in neuralset format.
        """
        sub = timeline["subject"]
        ses = timeline.get("session")
        run = timeline.get("run")
        task = timeline.get("task", self.TASK)

        # --- fMRI event row ---
        bold_img = self._load_raw(timeline)
        n_volumes: int = bold_img.shape[-1]
        # Derive TR from the NIfTI header pixdim[4]
        tr_s: float = float(bold_img.header.get_zooms()[3])  # type: ignore[index]
        if tr_s <= 0:
            logger.warning("TR could not be read from NIfTI header; defaulting to 1.0 s")
            tr_s = 1.0

        fmri_row: dict[str, tp.Any] = {
            "type": "Fmri",
            "start": 0.0,
            "duration": float(n_volumes) * tr_s,
            "frequency": 1.0 / tr_s,
            "filepath": str(
                _utils.bold_path(
                    self._fmriprep_dir, sub, task,
                    session=ses, run=run,
                    space=self.space, resolution=self.resolution,
                )
            ),
            "space": self.space,
        }

        # --- Stimulus / behavioural events ---
        stim_events = self._load_stimulus_events(timeline)

        all_rows = [pd.DataFrame([fmri_row])]
        if not stim_events.empty:
            all_rows.append(stim_events)

        return pd.concat(all_rows, ignore_index=True)
