# Research data supporting "Atomic cluster expansion potential for the Si–H system"

This repository supports the manuscript: "Atomic cluster expansion potential for the Si–H system", available on arXiv: https://arxiv.org/abs/2510.15633

## Models
We provide all the models generated and discussed in this study in the [models](./models) folder, including the models generated in both ablation studies, and the symmetric vs antisymmetric models.

## Data
The labelled datasets are available in the [datasets](./data/datasets) folder, both in `.xyz`  format and `.pkl.gzip` format.
A pickled dataframe is provides additional information, which can be loaded with `pandas` as follows:

```python
import pandas as pd
df=pd.read_pickle('./data/datasets/combined-to-Ite6-filtered-Fmag-50-Emax-0-min-Si-1.6-SiH-1.0-train.pkl.gzip',compression="gzip")
df.keys()
```

    Index(['energy', 'forces', 'ase_atoms', 'energy_corrected', 'atomic_env',
        'nnb', 'F_max', 'nb_atoms', 'e_corrected_per_atom', 'volume',
        'vol_per_atom', 'config', 'config_color', 'bulk', 'bulk_color', 'label',
        'ev_color', 'comp_dict', 'nH', 'cH', 'nSi', 'cSi', 'iteration',
        'force_mag', 'min_Si', 'min_H', 'min_SiH', 'min_dists'],
        dtype='object')


All data generated for the validation of the potential are available in the [paper](./data/paper) folder.

## Data analysis
We provide our scripts for the analysis and plotting of all figures in the manuscript in the [scripts](./scripts) folders.

### Supplemental structural analysis for a-Si:H
We provide supplemental real-space structural analysis of the a-Si:H structures for increasing H content in [this notebook](./src/scripts/SI_amorphous.ipynb).
