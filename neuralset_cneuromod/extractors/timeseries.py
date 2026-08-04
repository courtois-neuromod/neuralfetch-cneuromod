"""
Create new extractor class (tutorial)
https://facebookresearch.github.io/neuroai/neuralset/auto_examples/walkthrough/04_extractors.html

Reference: neuro (fMRI/MEG/spikes/EEG) extractors
https://github.com/facebookresearch/neuroai/blob/main/neuralset-repo/neuralset/extractors/neuro.py

https://github.com/facebookresearch/neuroai/blob/af8ce00c38aae77d4154c97714a12b2af332f1a1/neuralset-repo/neuralset/extractors/neuro.py#L1785

"""
from exca import MapInfra
from tqdm import tqdm
import typing as tp

import numpy as np
from neuralset import base, BaseExtractor
from neuralset.base import TimedArray as TimedArray

from neuralfetch_cneuromod.base import Timeseries


class TimeseriesExtractor(BaseExtractor):
    """fMRI timeseries extraction with optional caching to a NumPy memmap.

    Input: an HDF5 file with fmri timeseries of shape [n_voxels/n_parcels, time] nested per session 
    and per run for each subject.

    Optional **Temporal resampling** (``frequency``):
       If active, resamples the data to the target frequency, using np.interp.

    Parameters
    ----------
    offset : float
        Seconds to shift TRs forward to align delayed BOLD response.
    frequency : ``"native"`` | float
        Target sampling frequency.
    padding : int | ``"auto"`` | None
        Pad 1-D+T data to a uniform voxel count across subjects.
    query : Query | None
        Per-event predicate selecting which fMRI variant(s) to load, evaluated
        directly on ``Fmri`` event objects. This is a deliberate subset of the
        pandas ``QueryEvents`` dialect: it supports ``space``, ``preproc``, and
        ``study`` string conditions with ``==``, ``!=``, ``in``, and ``not in``
        combined with ``and``/``or``/``not``. Referenced names are validated at
        construction, so typos fail fast — including in short-circuited
        branches. For richer filters, use a
        ``QueryEvents`` transform upstream.
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

    query: base.Query | None = None
    infra: MapInfra = MapInfra(
        timeout_min=120,
        cpus_per_task=10,
        version="2",
    )

    def _preprocess_event(self, event: Timeseries) -> TimedArray:
        """"""
        rec = event.read()
        header: dict[str, tp.Any] = {"timeseries": event.timeseries, "space": event.space}
        return TimedArray(
            data=data.astype(np.float32),
            frequency=event.frequency,
            start=float("inf"),
            duration=event.duration,
            header=header,
        )
        
    def _get_data(self, events: list[Timeseries]) -> tp.Iterable[TimedArray]:
        """expensive per-event computation (typically cached via ``exca.MapInfra``)"""
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


