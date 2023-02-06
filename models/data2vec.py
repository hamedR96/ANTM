import torch
from transformers import Data2VecTextConfig, Data2VecTextModel,RobertaTokenizer

configuration = Data2VecTextConfig()
model = Data2VecTextModel(configuration).from_pretrained("facebook/data2vec-text-base")
tokenizer = RobertaTokenizer.from_pretrained("facebook/data2vec-text-base")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return embeddings[0].detach().numpy()

def data2vec_embedding(sentence):
    sentences_words = sentence.split()
    sentence_size = len(sentences_words)
    if 0 < sentence_size < 251:
        encoded_input = tokenizer(sentence, return_tensors='pt')
        model_output = model(**encoded_input)
        return mean_pooling(model_output, encoded_input["attention_mask"])
    elif sentence_size > 250:
        part1=data2vec_embedding( ' '.join(sentences_words[:250]))
        part2=data2vec_embedding(' '.join(sentences_words[250:]))
        return (part1+part2)/2
