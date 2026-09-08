from tqdm import tqdm
import typing as tp

import numpy as np
from neuralset import BaseExtractor
from neuralset.base import TimedArray as TimedArray

from neuralset_cneuromod.events.timeseries import Timeseries


class TimeseriesExtractor(BaseExtractor):
    """fMRI timeseries extraction (no caching).

    Input: an HDF5 file with fmri timeseries of shape [time, n_voxels/n_parcels] nested 
    per session and per run for each subject.

    Parameters
    ----------
    offset : float
        Seconds to shift TRs forward to align delayed BOLD response.
    frequency : ``"native"`` | float
        Target sampling frequency.
    """
    offset: float = 0.0
    event_types: tp.Literal["Timeseries"] = "Timeseries"
    aggregation: tp.Literal[
        "single",
        "sum",
        "mean",
        "first",
        "middle",
        "last",
        "cat",
        "stack",
        "trigger",
    ] = "single" 
    allow_missing: bool = False
    frequency: tp.Literal["native"] | float = "native"

    def _preprocess_event(self, event: Timeseries) -> TimedArray:
        """"""
        rec = event.read()
        header: dict[str, tp.Any] = {"timeseries": event.timeseries, "space": event.space}
        return TimedArray(
            data=rec.astype(np.float32),
            frequency=event.frequency,
            start=float("inf"),
            duration=event.duration,
            header=header,
        )
        
    def _get_data(self, events: list[Timeseries]) -> tp.Iterable[TimedArray]:
        """per-event computation (no caching)"""
        for event in tqdm(events, disable=len(events) < 2, desc="Processing timeseries data"):
            yield self._preprocess_event(event)

    def _get_timed_arrays(
        self, events: list[Timeseries], start: float, duration: float
    ) -> tp.Iterable[TimedArray]:
        """return an iterable of :class:`~neuralset.base.TimedArray`, one per event"""
        for event, ta in zip(events, self._get_data(events)):
            out = ta.copy(start=event.start - self.offset)
            out = out.overlap(start, duration)
            yield out


