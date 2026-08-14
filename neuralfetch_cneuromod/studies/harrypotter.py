"""Harry Potter book reading fMRI dataset.

Five subjects (sub-01, sub-02, sub-03, sub-05 and sub-06) read chapter 9 of the book
Harry Potter and the Philosopher's Stone (1997) while undergoing 3T fMRI.  

The text was presented word by word at a 2Hz pace (each word presented for 0.5s),
and split across 7 fMRI runs of ~9 to ~14 min in duration administered in a single session.

Stimuli are stored as time-stamped single words inside _events.tsv files in
the raw BIDS repository.

References
----------
* CNeuroMod documentation: https://docs.cneuromod.ca/latest/datasets/harrypotter.html
* DataLad BIDS repo: https://github.com/courtois-neuromod/harrypotter
* DataLad fMRIPrep repo: https://github.com/courtois-neuromod/harrypotter.fmriprep
* DataLad timeseries repo: https://github.com/courtois-neuromod/harrypotter.timeseries
"""

from __future__ import annotations

import typing as tp
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from neuralfetch_cneuromod import _utils
from neuralfetch_cneuromod.base import CNeuroModStudy


def get_bold_runs(
    fmriprep_dir: Path,
    subject: str,
    *,
    space: str = _utils.DEFAULT_SPACE,
) -> list[tuple[str, str | None, Path]]:
    """Return sorted list of run labels for which a preproc BOLD file exists.

    Parameters
    ----------
    fmriprep_dir:
        Root of the fMRIPrep derivatives dataset.
    subject:
        Subject label without ``sub-`` prefix.
    space:
        fMRIPrep output space template.

    Returns
    -------
    list[tuple[str, str | None, Path]]
        - Task run label (without ``task-`` prefix), 
        - Run number (without ``run-`` prefix),
        - Full path to run BOLD file
    """
    sub_dir = fmriprep_dir / f"sub-{subject}" / "func"

    if not sub_dir.exists():
        return []

    pattern = f"sub-{subject}_task-*_space-{space}_desc-preproc_bold.nii.gz"
    bold_files = sorted(sub_dir.glob(pattern))
    runs = []
    for f in bold_files:
        # Extract task-run entity from filename
        stem = f.name
        run: str | None = None
        task: str | None = None        
        for entity in stem.split("_"):
            if entity.startswith("task-"):
                task = entity[5:]
            elif entity.startswith("run-"):
                run = entity[4:]
                break
        runs.append((task, run, f))

    return runs if runs else []


def iter_bids_runs(
    fmriprep_dir: Path,
    *,
    subjects: list[str] | None = None,
    space: str = _utils.DEFAULT_SPACE,
) -> Iterator[dict[str, Any]]:
    """Iterate over all available (subject, run) pairs in *fmriprep_dir*.

    Only pairs for which a preprocessed BOLD file actually exists on disk
    are yielded.  This iterator called by ``iter_timelines()`` is customized 
    for this study classes because the dataset lacks session numbers.

    Parameters
    ----------
    fmriprep_dir:
        Root of the fMRIPrep derivatives dataset.
    subjects:
        Restrict iteration to these subject labels (without ``sub-`` prefix).
        Defaults to all subjects found in *fmriprep_dir*.
    space:
        fMRIPrep output space template.

    Yields
    ------
    dict
        Keys: ``subject`` (str), ``file_path`` (str), ``session`` (str), ``task`` (str),
        ``run`` (str | None).
    """
    available_subjects = _utils.get_subjects(fmriprep_dir)
    if subjects is not None:
        available_subjects = [s for s in available_subjects if s in subjects]

    for sub in available_subjects:
        runs = get_bold_runs(
            fmriprep_dir, sub, space=space,
        )
        for task, run, run_path in runs:
            yield dict(
                subject=sub, file_path=str(run_path),
                session=None, task=task, run=run,
            )


class HarryPotter(CNeuroModStudy):
    """Courtois NeuroMod — *Harry Potter* audiobook listening dataset.

    Five subjects read chapter 9 of Harry Potter and the Philosopher's Stone
    (1997) during 3T fMRI.  Each BOLD run covers one segment of the chapter.

    Parameters
    ----------
    path:
        Root data directory.  Resolves ``{path}/harrypotter/bids`` and
        ``{path}/harrypotter/fmriprep`` or ``{path}/harrypotter/timeseries``,
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
    The BIDS events TSV contains word-level onsets derived from forced
    alignment of the audiobook text with the audio stimulus.

    Examples
    --------
    >>> study = HarryPotter(path="path/to/cneuromod.all")
    >>> events = study.run()
    """

    TASK: tp.ClassVar[str] = "harrypotter"
    BIDS_REPO: tp.ClassVar[str] = "harrypotter"
    FMRIPREP_REPO: tp.ClassVar[str] = "harrypotter.fmriprep"
    TIMESERIES_REPO: tp.ClassVar[str] = "harrypotter.timeseries"

    dataset_name: tp.ClassVar[str] = "CNeuroMod HarryPotter"
    description: tp.ClassVar[str] = (
        "Five subjects listening to the Harry Potter audiobook (chapters 1-9) "
        "during 3T fMRI."
    )
    bibtex: tp.ClassVar[str] = CNeuroModStudy.bibtex

    # -----------------------------------------------------------------
    # Download pattern builders
    # -----------------------------------------------------------------

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
                f"{self.path}/bids/{sub}/func/{sub}_task-*_events.tsv",
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
                f"{self.path}/fmriprep/{sub}/func/{sub}_"
                f"task-*_{space}_desc-preproc_bold.nii.gz",
                
                # Confound regressors
                f"{self.path}/fmriprep/{sub}/func/{sub}_"
                f"task-*_desc-confounds_timeseries.tsv",
                
                # Brain mask in the target space
                f"{self.path}/fmriprep/{sub}/func/{sub}_"
                f"task-*_{space}_desc-brain_mask.nii.gz",
            ])
        return patterns

    # -----------------------------------------------------------------
    # Timeline iteration
    # -----------------------------------------------------------------

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Iterate over all available (subject, session, run) triples.

        * **BOLD** — Uses :meth:`iter_bids_runs` to discover
        available preprocessed BOLD files in the fMRIPrep directory.
        * **Timeseries** — Uses :func:`~neuralfetch_cneuromod._utils.iter_tseries_runs` 
        to discover pre-extracted timeseries nested in .h5 files in the timeseries directory.

        Yields
        ------
        dict
            Keys: ``subject`` (str), ``session`` (str | None), ``file_path`` (str), 
            ``task`` (str | None), ``run`` (str | None).

        Raises
        ------
        FileNotFoundError
            If :attr:`fmriprep_dir` does not exist.
        """
        if self.timeseries is None:
            yield from iter_bids_runs(
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

    def _load_stimulus_events(
        self, timeline: dict[str, tp.Any], event_root: str,
    ) -> pd.DataFrame:
        """Load word events for *timeline*.

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

        # run num: fmriprep/tseries -> bids
        e_root = event_root.replace("run-", "run-0").replace("ses-001_", "")
        ep_list = sorted(glob.glob(
            f"{self._bids_dir}/sub-{timeline['subject']}"
            f"/func/{event_root}*events.tsv"
        ))
        if len(ep_list) != 1:
            self.logger.debug("No unique events file found: %s", ep_list[0])
            return pd.DataFrame()

        bids_events = pd.read_csv(ep_list[0], sep="\t")
        # Map BIDS columns to neuralset conventions
        word_events = []
        for _, row in bids_events.iterrows():
            word_events.append(
                {
                    "type": "Word",
                    "text": row["word"],
                    "start": row["onset"],
                    "stop": row["onset"] + row["duration"],
                    "duration": row["duration"],
                    "language": "en",
                    "modality": "read",
                }
            )
        return pd.DataFrame(word_events)