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

## User Install

```bash
pip install neuralfetch-cneuromod
# With DataLad download support:
pip install "neuralfetch-cneuromod[datalad]"
# With full extras (datalad, bids layout, nilearn):
pip install "neuralfetch-cneuromod[all]"
```

## Developer Install

```bash
# Clone the repo locally
git clone git@github.com:courtois-neuromod/neuralfetch-cneuromod.git
# Install packages in editable mode
pip install -e neuralfetch-cneuromod[all,dev]
```


## Accessing the CNeuroMod Dataset

CNeuroMod data is organized as a datalad super-dataset with one submodule per study/derivative. 

**CNeuroMod Data Structure**
```
cneuromod.all/
├── friends/
│   ├── bids/          ← raw BIDS (MRI, events, stimuli)
│   ├── fmriprep/      ← fMRIPrep derivatives (preprocessed BOLD)
│   └── timeseries/    ← masked and denoised BOLD timeseries
├── harrypotter/
│   ├── bids/
│   ├── fmriprep/
│   └── timeseries/
└── ...
```

Use the [DataLad software](https://www.datalad.org/) (see **User / Developer Install**) to clone the super-dataset repository from Github. This step only downloads symbolic links used to retrieve the data files (no large files are downloaded).

```bash
datalad clone git@github.com:courtois-neuromod/cneuromod.all.git
```
A warning message will be thrown because the remote origin does not have git-annex installed. This issue will not prevent the installation.

Download symbolic links to data files stored in submodules nested per study. Navigate to a study folder, and use `datalad get` to pull links per submodule (no large files downloaded).

E.g.,
```bash
cd cneuromod.all/friends
datalad get bids/*
datalad get fmriprep/*
```

Use the Study classes to perform a selective download of the files you need for modelling with `datalad get`. The downloading step takes a long time, but it only needs to be performed once when you first instantiate a new Study class.  


## Quick Start


```python
from neuralfetch_cneuromod.studies.friends import Friends

# The study path can point to the cloned cneuromod.all repository (RECOMMENDED), 
# to a specific study subfolder (e.g., `cneuromod.all/friends`). 

# Alternatively, it can point to a folder that shares the study name (e.g., `/path/to/friends`). If that folder is empty or non-existent, the study class will
# use DataLad to clone and pull from the proper set of repositories.

# In either scenario, the study class resolves its subfolder structure 
# automatically. 

study = Friends(path="path/to/cneuromod.all")
print(study.study_summary())

# By default, the Study class tracks fMRI data processed with fMRIprep.
# Use the `modalities` parameter to track pre-masked, pre-denoised fMRI timeseries.
# e.g., study = Friends(path="path/to/cneuromod.all", modalities=['timeseries', 'events'])

# Load all events as a neuralset-compatible DataFrame
# This step uses `datalad get` to download files selectively, 
# i.e., the first attempt is much slowed than subsequent ones
events = study.run()

# Optionally, you can pre-download data files as a separate step 
# (requires SSH key + access) before `study.run()`:
study.download()
```


## License

MIT. The CNeuroMod datasets themselves are CC0 (open subjects 01, 03, 05) or require a
data transfer agreement for all 6 subjects. See https://www.cneuromod.ca/ for details.

## Citation

If you use CNeuroMod data, please include:

> The Courtois project on neural modelling was made possible by a generous donation from the
> Courtois foundation, administered by the Fondation Institut Gériatrie Montréal at CIUSSS du
> Centre-Sud-de-l'île-de-Montréal and the University of Montreal. See https://docs.cneuromod.ca.
