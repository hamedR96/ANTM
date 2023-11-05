import torch

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return embeddings[0].detach().cpu().numpy()


def data2vec_embedding(sentence, tokenizer, model,summarizer,device):
    encoded_input = tokenizer(sentence, return_tensors='pt').to(device)
    # Pass the input through the model
    try:
        model_output = model(**encoded_input)
        return mean_pooling(model_output, encoded_input["attention_mask"])
    except:
        print("Summarizing a document with BART due to its Large length for Embedding...")
        new_sentences=summarizer(sentence, max_length=512, do_sample=False)[0]["summary_text"]
        return(data2vec_embedding(new_sentences,tokenizer, model,summarizer,device))

