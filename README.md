# neuralfetch-cneuromod

**NeuralFetch** study classes for the [Courtois NeuroMod](https://www.cneuromod.ca/) dense fMRI dataset.

Extends the [NeuroAI NeuralFetch](https://github.com/facebookresearch/neuroai) framework to fetch and
load fMRIPrep-preprocessed BOLD fMRI and associated stimuli for all CNeuroMod datasets via
[DataLad](https://www.datalad.org/).

## Datasets

| Study class | Dataset | Task |
|---|---|---|
| `Friends` | Friends TV show seasons 1-6 | Audio-visual movie |
| `HarryPotter` | Harry Potter audiobook | Auditory narrative |
| `Mario` | Super Mario Bros gameplay | Video game |
| `Shinobi` | Shinobi video game | Video game |
| `Movie10` | 10 open-access Hollywood movies | Audio-visual movie |
| `Things` | THINGS object image set | Visual object recognition |
| `Floc` | Functional localizer (fLoc) | Visual categories |
| `HcpTrt` | HCP-style test-retest | Multimodal |
| `Retinotopy` | Population receptive fields | Visual retinotopy |
| `Gamepad` | Gamepad motor task | Motor |
| `CNeuroModAnat` | Anatomy / structural MRI | — |

## Install

```bash
pip install neuralfetch-cneuromod
# With DataLad download support:
pip install "neuralfetch-cneuromod[datalad]"
# Full extras:
pip install "neuralfetch-cneuromod[all]"
```

## Quick Start

```python
from neuralfetch_cneuromod.studies.friends import Friends

# Point to the root folder that contains your CNeuroMod DataLad repos.
# The study expects:
#   /data/cneuromod/friends/bids/     (raw BIDS data)
#   /data/cneuromod/friends/fmriprep/ (fMRIPrep derivatives)

study = Friends(path="/data/cneuromod")
print(study.study_summary())

# Load all events as a neuralset-compatible DataFrame
events = study.run()

# Optionally download the DataLad repos first (requires SSH key + access):
study.download()
```

## CNeuroMod Data Structure

CNeuroMod data is organized as a datalad super-dataset with one submodule per study/derivative:

```
cneuromod.all/
├── friends/
│   ├── bids/          ← raw BIDS (MRI, events, stimuli)
│   └── fmriprep/      ← fMRIPrep derivatives
├── harrypotter/
│   ├── bids/
│   └── fmriprep/
└── ...
```

Each study class reads from `{path}/{study_name}/bids` and `{path}/{study_name}/fmriprep`.
The `path` can point directly to the study root or to the super-dataset root — the class resolves
its subfolder automatically.

## License

MIT. The CNeuroMod datasets themselves are CC0 (open subjects 01, 03, 05) or require a
data transfer agreement for all 6 subjects. See https://www.cneuromod.ca/ for details.

## Citation

If you use CNeuroMod data, please include:

> The Courtois project on neural modelling was made possible by a generous donation from the
> Courtois foundation, administered by the Fondation Institut Gériatrie Montréal at CIUSSS du
> Centre-Sud-de-l'île-de-Montréal and the University of Montreal. See https://docs.cneuromod.ca.
