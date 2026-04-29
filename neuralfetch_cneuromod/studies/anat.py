"""Anatomical MRI dataset (anat).

This module provides access to the CNeuroMod structural MRI derivatives:
sMRIPrep and FreeSurfer outputs for the six subjects.  No fMRI data is
exposed — the study enumerates subjects and exposes T1w NIfTI and FreeSurfer
parcellation paths.

References
----------
* CNeuroMod documentation: https://docs.cneuromod.ca/en/latest/DATASETS.html#anat
* DataLad BIDS repo: https://github.com/courtois-neuromod/anat
* DataLad derivative repo: https://github.com/courtois-neuromod/anat.smriprep
"""

from __future__ import annotations

import logging
import typing as tp
from pathlib import Path

import pandas as pd
import pydantic

from neuralfetch.download import Datalad
from neuralset.events import study as _study

from neuralfetch_cneuromod import _utils

logger = logging.getLogger(__name__)

_CNEUROMOD_GH = "https://github.com/courtois-neuromod/{repo}.git"


def _repo_name(url: str) -> str:
    """Extract the repository name from a GitHub URL, stripping the ``.git`` suffix."""
    name = Path(url).name
    return name[:-4] if name.endswith(".git") else name


class CNeuroModAnat(_study.Study):
    """Courtois NeuroMod — anatomical MRI dataset.

    Exposes T1w NIfTI files and sMRIPrep/FreeSurfer derivative paths for each
    CNeuroMod subject.  This study does not contain fMRI data; each
    *timeline* corresponds to one subject's anatomical session.

    Parameters
    ----------
    path:
        Root data directory.  Resolves ``{path}/CNeuroModAnat/bids`` and
        ``{path}/CNeuroModAnat/smriprep``.
    subjects:
        Restrict to a subset of subjects (labels without ``sub-`` prefix).
        Defaults to all available subjects.
    datalad_jobs:
        Parallel DataLad download jobs.

    Notes
    -----
    The ``bids/`` submodule contains the raw T1w images (one or more per
    subject / session).  The ``smriprep/`` derivative contains brain-extracted
    and MNI-normalised T1w images, brain masks, and FreeSurfer parcellations.

    Examples
    --------
    >>> study = CNeuroModAnat(path="/data/cneuromod")
    >>> print(study.study_summary())
    """

    BIDS_REPO: tp.ClassVar[str] = "anat"
    SMRIPREP_REPO: tp.ClassVar[str] = "anat.smriprep"

    dataset_name: tp.ClassVar[str] = "CNeuroMod Anat"
    description: tp.ClassVar[str] = (
        "T1w structural MRI for six CNeuroMod subjects with sMRIPrep derivatives."
    )
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

    # Pydantic fields
    subjects: list[str] | None = None
    datalad_jobs: int = 1

    _bids_dir: Path = pydantic.PrivateAttr(default=None)  # type: ignore[assignment]
    _smriprep_dir: Path = pydantic.PrivateAttr(default=None)  # type: ignore[assignment]

    def model_post_init(self, log__: tp.Any) -> None:
        """Resolve bids/ and smriprep/ directories after init."""
        super().model_post_init(log__)
        bids_url = _CNEUROMOD_GH.format(repo=self.BIDS_REPO)
        smriprep_url = _CNEUROMOD_GH.format(repo=self.SMRIPREP_REPO)
        self._bids_dir = self._resolve_anat_subdir("bids", _repo_name(bids_url))
        self._smriprep_dir = self._resolve_anat_subdir("smriprep", _repo_name(smriprep_url))

    def _resolve_anat_subdir(self, folder: str, repo_name: str) -> Path:
        """Resolve a DataLad-managed or manually cloned sub-repository path.

        Tries the Datalad-managed layout (``path/folder/repo_name``) first;
        falls back to the flat layout (``path/folder``).

        Parameters
        ----------
        folder:
            Sub-directory name (``"bids"`` or ``"smriprep"``).
        repo_name:
            Repository name without the ``.git`` suffix.

        Returns
        -------
        Path
            Resolved path (may not yet exist on disk).
        """
        datalad_managed = self.path / folder / repo_name
        if datalad_managed.exists():
            return datalad_managed
        return self.path / folder

    @property
    def bids_dir(self) -> Path:
        """Path to the raw BIDS DataLad repository (``{path}/bids``)."""
        return self._bids_dir

    @property
    def smriprep_dir(self) -> Path:
        """Path to the sMRIPrep derivative DataLad repository (``{path}/smriprep``)."""
        return self._smriprep_dir

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def _download(self) -> None:
        """Clone and fetch the anatomical BIDS and sMRIPrep DataLad repos.

        Delegates to :class:`neuralfetch.download.Datalad` for both the raw
        BIDS dataset (clone only) and the sMRIPrep derivative (clone + full
        ``datalad get``).  Idempotency is handled by success files.

        Raises
        ------
        RuntimeError
            If DataLad is not installed or the operations fail.
        """
        bids_url = _CNEUROMOD_GH.format(repo=self.BIDS_REPO)
        smriprep_url = _CNEUROMOD_GH.format(repo=self.SMRIPREP_REPO)

        # Raw BIDS: clone only (no annexed content retrieval)
        bids_dl = Datalad(
            study="CNeuroModAnat_bids",
            dset_dir=self.path,
            folder="bids",
            repo_url=bids_url,
            threads=self.datalad_jobs,
            folders=[],
        )
        logger.info("Cloning anat BIDS repo → %s/bids", self.path)
        bids_dl.download()

        # sMRIPrep: clone + retrieve all derivative content
        smriprep_dl = Datalad(
            study="CNeuroModAnat_smriprep",
            dset_dir=self.path,
            folder="smriprep",
            repo_url=smriprep_url,
            threads=self.datalad_jobs
        )
        logger.info("Cloning + fetching sMRIPrep repo → %s/smriprep", self.path)
        smriprep_dl.download()

        # Re-resolve directory pointers
        self._bids_dir = self._resolve_anat_subdir("bids", bids_dl.repo_name)
        self._smriprep_dir = self._resolve_anat_subdir("smriprep", smriprep_dl.repo_name)

    # ------------------------------------------------------------------
    # Timeline iteration
    # ------------------------------------------------------------------

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Iterate over subjects with available T1w files in the BIDS directory.

        Each timeline corresponds to one subject (one row in the summary).

        Yields
        ------
        dict
            Keys: ``subject`` (str), with value being the subject label
            without the ``sub-`` prefix.

        Raises
        ------
        FileNotFoundError
            If neither the BIDS nor the sMRIPrep directory exists.
        """
        search_dir = self._bids_dir if self._bids_dir.exists() else self._smriprep_dir
        if not search_dir.exists():
            raise FileNotFoundError(
                f"Neither bids ({self._bids_dir}) nor smriprep ({self._smriprep_dir}) "
                "directories found.  Run study.download() first."
            )

        available = _utils.get_subjects(search_dir)
        if self.subjects is not None:
            available = [s for s in available if s in self.subjects]

        for sub in available:
            yield dict(subject=sub)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def t1w_path(self, subject: str) -> Path:
        """Return the sMRIPrep brain-extracted T1w NIfTI for *subject*.

        Parameters
        ----------
        subject:
            Subject label without the ``sub-`` prefix.

        Returns
        -------
        Path
            Expected path to the MNI-space T1w file in sMRIPrep derivatives.
        """
        return (
            self._smriprep_dir
            / "smriprep"
            / f"sub-{subject}"
            / "anat"
            / f"sub-{subject}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz"
        )

    def brain_mask_path(self, subject: str) -> Path:
        """Return the sMRIPrep brain mask NIfTI for *subject*.

        Parameters
        ----------
        subject:
            Subject label without the ``sub-`` prefix.

        Returns
        -------
        Path
            Expected path to the MNI-space brain mask file.
        """
        return (
            self._smriprep_dir
            / "smriprep"
            / f"sub-{subject}"
            / "anat"
            / f"sub-{subject}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz"
        )

    # ------------------------------------------------------------------
    # Event loading
    # ------------------------------------------------------------------

    def _load_timeline_events(
        self, timeline: dict[str, tp.Any]
    ) -> pd.DataFrame:
        """Return a single-row events DataFrame describing the T1w file for *subject*.

        Parameters
        ----------
        timeline:
            Timeline dict with ``subject`` key.

        Returns
        -------
        pd.DataFrame
            One-row events table with columns ``type``, ``start``,
            ``duration``, ``filepath``.
        """
        sub = timeline["subject"]
        t1w = self.t1w_path(sub)
        return pd.DataFrame(
            [
                {
                    "type": "T1w",
                    "start": 0.0,
                    "duration": 0.0,
                    "filepath": str(t1w),
                    "space": "MNI152NLin2009cAsym",
                }
            ]
        )
