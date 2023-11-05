from antm.data2vec import data2vec_embedding
from sentence_transformers import SentenceTransformer
from transformers import  Data2VecTextModel, RobertaTokenizer
from transformers import pipeline
import swifter


def contextual_embedding(df,mode,device):
    if mode=="bert":
        df["embedding"] = list(SentenceTransformer('all-mpnet-base-v2',device=device).encode(df.content.to_list(),show_progress_bar=True))
    elif mode=="data2vec":
        tokenizer = RobertaTokenizer.from_pretrained("facebook/data2vec-text-base")
        model = Data2VecTextModel.from_pretrained("facebook/data2vec-text-base").to(device)
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        df["embedding"] = df["content"].swifter.apply(lambda x: data2vec_embedding(x,tokenizer, model, summarizer,device))
    else:
        df["embedding"] = list(SentenceTransformer('all-MiniLM-L6-v2',device=device).encode(df.content.to_list(),show_progress_bar=True))
    return df
