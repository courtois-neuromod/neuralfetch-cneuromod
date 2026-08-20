
from __future__ import annotations

import typing as tp

import h5py
import numpy as np
import pandas as pd

from neuralset.base import StrCast, Frequency
from neuralset.events import etypes

from neuralfetch_cneuromod import _utils


class Timeseries(etypes.BaseSplittableEvent):
    """Pre-processed, masked, detrended and normalized functional MRI (fMRI) 
    recording event.

    Requires :code:`h5py` to be installed.

    Supports chunking via read() so chunks load only their own slice.

    Parameters
    ----------
    subject : str
        Subject identifier, e.g. ``"01"`` (required).
    filepath : Path or str
        Path to the .HDF5 file containing nested timeseries.     
    session : str
        Session identifier, e.g. ``"ses-001"`` (required). First-level key
        in .HDF5 file structure.
    run : str
        Run identifier, e.g. ``"ses-001_task-bourne01_timeseries"`` (required).
        Second-level key in .HDF5 file structure.
    frequency : float
        Sampling frequency in Hz (required).
    timeseries : str
        Timeseries format, e.g. ``"cneuromod2026"``, ``"schaefer1000"``,
        ``"voxel_mni"``, ``"voxel_native"``.
    space : str
        Coordinate space before timeseries extraction,
        e.g. ``"MNI152NLin2009cAsym"``, ``"T1w"``.
    """
    subject: StrCast
    session: str | None = None
    run: str | None = None
    timeseries: str = _utils.DEFAULT_TIMESERIES
    space: str = _utils.DEFAULT_SPACE

    def model_post_init(self, log__: tp.Any) -> None:
        if not self.frequency or pd.isna(self.frequency):
            raise ValueError(
                "Frequency must be provided for Timeseries event."
            )
        if not self.duration:
            raise ValueError(
                "Duration must be provided for Timeseries event."
            )
        if not self.session:
            raise ValueError(
                "Session must be provided for Timeseries event."
            )
        if not self.run:
            raise ValueError("Run must be provided for Timeseries event.")
        super().model_post_init(log__)

    def read(self) -> tp.Any:
        # If need be, crop based on specified offser and duration``.
        tseries = super().read()
        sr = Frequency(self.frequency)
        start_vol = sr.to_ind(self.offset)
        end_vol = start_vol + sr.to_ind(self.duration)
        if start_vol == 0 and end_vol >= tseries.shape[0]:
            return tseries
        return tseries[:, start_vol:end_vol]  # chunked

    def _read(self) -> tp.Any:
        with h5py.File(self.filepath, "r") as f:
            tseries = np.array(f[self.session][self.run]).T  # TimedArray last dim is time when freq > 0
        return tseries