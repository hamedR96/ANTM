import torch
from transformers import Data2VecTextConfig, Data2VecTextModel,RobertaTokenizer
from transformers import pipeline

configuration = Data2VecTextConfig()
model = Data2VecTextModel(configuration).from_pretrained("facebook/data2vec-text-base")
tokenizer = RobertaTokenizer.from_pretrained("facebook/data2vec-text-base",model_max_length=1024)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return embeddings[0].detach().numpy()


def data2vec_embedding(sentence):
    encoded_input = tokenizer(sentence, return_tensors='pt')
    # Pass the input through the model
    try:
        model_output = model(**encoded_input)
        return mean_pooling(model_output, encoded_input["attention_mask"])
    except:
        print("Summarizing a document with BART due to its Large length for Embedding...")
        new_sentences=summarizer(sentence, max_length=512, do_sample=False)[0]["summary_text"]
        return(data2vec_embedding(new_sentences))
