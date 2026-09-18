# Copyright (c) 2026 Basile Pinsard
# MIT License — see LICENSE for details.

"""neuralfetch-cneuromod: NeuralFetch study classes for the Courtois NeuroMod dataset.

This package extends the NeuroAI NeuralFetch library to provide Study subclasses for
all Courtois NeuroMod (CNeuroMod) datasets. It registers itself as a ``neuralset.studies``
entry-point so that study discovery is automatic upon installation:

    >>> from neuralset.events.study import Study
    >>> catalog = Study.catalog()
    >>> "Friends" in catalog
    True

Data is fetched via DataLad from the courtois-neuromod GitHub organization and loaded from
fMRIPrep-preprocessed BIDS derivatives or pre-extracted BOLD timeseries.
"""

from neuralfetch_cneuromod import studies  # noqa: F401 — ensures studies are registered

__version__ = "0.1.0"
__all__ = ["studies"]
