from models.data2vec import data2vec_embedding
from sentence_transformers import SentenceTransformer
import swifter

def contextual_embedding(df,mode):
    if mode=="data2vec":
        df["embedding"]=df["abstract"].swifter.apply(lambda x: data2vec_embedding(x))
    else:
        df["embedding"] = list(SentenceTransformer('all-mpnet-base-v2').encode(df.abstract.to_list()))
    return df