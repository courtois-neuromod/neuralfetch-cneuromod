"""CNeuroMod study modules.

All concrete study classes are imported here so that the neuralset
entry-point scanner registers them when it imports this package.
"""

from neuralfetch_cneuromod.studies.anat import CNeuroModAnat  # noqa: F401
from neuralfetch_cneuromod.studies.floc import Floc  # noqa: F401
from neuralfetch_cneuromod.studies.friends import Friends  # noqa: F401
from neuralfetch_cneuromod.studies.gamepad import Gamepad  # noqa: F401
from neuralfetch_cneuromod.studies.harrypotter import HarryPotter  # noqa: F401
from neuralfetch_cneuromod.studies.hcptrt import HcpTrt  # noqa: F401
from neuralfetch_cneuromod.studies.mario import Mario  # noqa: F401
from neuralfetch_cneuromod.studies.movie10 import Movie10  # noqa: F401
from neuralfetch_cneuromod.studies.retinotopy import Retinotopy  # noqa: F401
from neuralfetch_cneuromod.studies.shinobi import Shinobi  # noqa: F401
from neuralfetch_cneuromod.studies.things import Things  # noqa: F401

__all__ = [
    "CNeuroModAnat",
    "Floc",
    "Friends",
    "Gamepad",
    "HarryPotter",
    "HcpTrt",
    "Mario",
    "Movie10",
    "Retinotopy",
    "Shinobi",
    "Things",
]
