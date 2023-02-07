
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://hamedrahimi.fr)
[![arXiv](https://img.shields.io/badge/arXiv-2302.01501-<COLOR>.svg)](https://arxiv.org/abs/2302.01501)

# ANTM
ANTM: An Aligned Neural Topic Model for Exploring Evolving Topics


 Dynamic topic models are effective methods that primarily focus on studying the evolution of topics present in a collection of documents. These models are widely used for understanding trends, exploring public opinion in social networks, or tracking research progress and discoveries in scientific archives. Since topics are defined as clusters of semantically similar documents, it is necessary to observe the changes in the content or themes of these clusters in order to understand how topics evolve as new knowledge is discovered over time. Here, we introduce a dynamic neural topic model called ANTM, which uses document embeddings (data2vec) to compute clusters of semantically similar documents at different periods, and aligns document clusters to represent their evolution. This alignment procedure preserves the temporal similarity of document clusters over time and captures the semantic change of words characterized by their context within different periods. Experiments on four different datasets show that ANTM outperforms probabilistic dynamic topic models (e.g. DTM, DETM) and significantly improves topic coherence and diversity over other existing dynamic neural topic models (e.g. BERTopic).


## Installation

Installation can be done using:

```bash
pip install requirements.txt
```

## Quick Start
As implemented in the notebook, We start extracting evolving topics from DBLP dataset containing computer science articles:

```python
import pandas as pd
from models.antm import ANTM

df=pd.read_parquet("./data/dblpFullSchema_2000_2020_extract_big_data_10K.parquet")
df=df[["abstract","year"]].dropna().reset_index()

window_size=3
overlap=1

evolving_topics=ANTM(df,overlap,window_size,mode="data2vec",num_words=10)
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
