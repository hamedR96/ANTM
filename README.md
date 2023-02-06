

[![arXiv](https://img.shields.io/badge/arXiv-2302.01501-<COLOR>.svg)](https://arxiv.org/abs/2302.01501)

# ANTM
ANTM: An Aligned Neural Topic Model for Exploring Evolving Topics


## Installation

Installation can be done using:

```bash
pip install requirements.txt
```

## Quick Start
As implemented in the notebook, We start extracting evolving topics from DBLP dataset containing computer science articles:

```python
import pandas as pd
df=pd.read_parquet("./data/dblpFullSchema_2000_2020_extract_big_data_10K.parquet")
df=df[["abstract","year"]].dropna().reset_index()

from models.antm import ANTM

window_size=3
overlap=1
evolving_topics=ANTM(df,overlap,window_size,mode="data2vec",umap_dimension_size=5,umap_n_neighbors=20,partioned_clusttering_size=5,num_words=10)
```


## Citation
To cite the [ANTM_Paper](https://arxiv.org/abs/2302.01501), please use the following bibtex reference:
```bibtext
@misc{rahimi2023antm,
      title={ANTM: An Aligned Neural Topic Model for Exploring Evolving Topics}, 
      author={Hamed Rahimi and Hubert Naacke and Camelia Constantin and Bernd Amann},
      year={2023},
      eprint={2302.01501},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```
