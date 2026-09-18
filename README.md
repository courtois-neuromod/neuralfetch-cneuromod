# neuralfetch-cneuromod

**NeuralFetch** study classes for the [Courtois NeuroMod](https://www.cneuromod.ca/) dense fMRI dataset.

Extends the [NeuroAI NeuralFetch](https://github.com/facebookresearch/neuroai) framework to fetch and
load fMRIPrep-preprocessed BOLD fMRI and associated stimuli for all CNeuroMod datasets via
[DataLad](https://www.datalad.org/).

## Datasets

| Study class | Dataset | Task |
|---|---|---|
| `Emotion-videos` | Cohen & Keltner emotion-evoking videos | Visual movie |
| `Floc` | Functional localizer (fLoc) | Semantic image categories |
| `Friends` | Friends TV show seasons 1-6 | Audio-visual movie |
| `HarryPotter` | Harry Potter book chapter | Written narrative |
| `HcpTrt` | HCP-style test-retest | Multimodal |
| `Mario` | Super Mario Bros gameplay | Video game |
| `Mario3` | Super Mario Bros 3 gameplay | Video game |
| `MarioStars` | Super Mario All-Stars gameplay | Video game |
| `Movie10` | 3 Hollywood movies and 1 BBC documentary | Audio-visual movie |
| `Narratives` | Stories from Nastase stimulus set | Auditory narrative and free recall |
| `OOD` | OOD movie stimuli | Audio-visual movie |
| `PetitPrince` | Le Petit Prince audiobook | Auditory narrative |
| `Retinotopy` | HCP retinotopy stimuli | Visual retinotopy |
| `Shinobi` | Shinobi video game | Video game |
| `Things` | THINGS object image set | Visual object recognition |
| `Triplets` | Set of concrete written words | Word similarity judgement |

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
│   ├── timeseries/    ← masked and denoised BOLD timeseries
│   ├── stimuli/       ← Friends episodes (.mkv) split per run
│   └── annotations/   ← Friends episodes annotations, e.g., transcripts
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
datalad get stimuli/*
datalad get annotations/*
```

Use the Study classes to download of the files you need for modelling (see Quick Start). Study.download() uses the [python DataLad API](https://docs.datalad.org/en/stable/generated/datalad.api.get.html) internally to perform a selective file download. The downloading step takes a long time, but it only needs to be performed once when you first instantiate a new Study class.  


## Quick Start

```python
from neuralfetch_cneuromod.studies.friends import Friends

study = Friends(path="path/to/cneuromod.all")
```

The study path can point to the cloned cneuromod.all repository (RECOMMENDED), or to a specific study subfolder inside cneuromod.all (e.g., `path/to/cneuromod.all/friends`). 

Alternatively, it can point to any folder that shares the study name (e.g., `/path/to/friends`). If that folder is empty or non-existent, the study class will use DataLad to clone and pull from the proper set of repositories.

In either scenario, the study class resolves its subfolder structure automatically. 

```python
# Download study files (requires SSH key + access).
# This step uses the `datalad get` command to download files selectively,
# which can be slow. Consider running inside a tmux session.
study.download()
print(study.study_summary())

# Load all events as a neuralset-compatible DataFrame
events = study.run()
```

By default, the Study class tracks fMRI data pre-processed with fMRIprep. Use the `timeseries` parameter to track pre-masked, pre-denoised fMRI timeseries instead.
E.g., 

```python
from neuralfetch_cneuromod.studies.friends import Friends

study = Friends(path="path/to/cneuromod.all", timeseries='cneuromod2026')

study.download()
events = study.run()
```

To speed up future iterations, you can use caching to cache the study's events DataFrame and its timelines.

E.g., 

```python
from pathlib import Path
from neuralfetch_cneuromod.studies.friends import Friends
CACHE = Path('/path/to/cache')

study = Friends(
    path="path/to/cneuromod.all",
    timeseries='cneuromod2026',
    infra={"folder": CACHE},
)

study.download()
events = study.run()
```

## License

MIT. The CNeuroMod datasets themselves are CC0 (open subjects 01, 02, 03, 05, 06) or require a data transfer agreement for all 6 subjects. See https://www.cneuromod.ca/ for details.

## Citation

If you use CNeuroMod data, please include:

> The Courtois project on neural modelling was made possible by a generous donation from the
> Courtois foundation, administered by the Fondation Institut Gériatrie Montréal at CIUSSS du
> Centre-Sud-de-l'île-de-Montréal and the University of Montreal. See https://docs.cneuromod.ca.
