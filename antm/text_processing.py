from collections import defaultdict
from nltk import word_tokenize, WordNetLemmatizer
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
import re


def preprocessing_documents(document_list):
    cleaned_documents=[doc.lower() for doc in document_list]
    cleaned_documents = [doc.replace("\n", " ") for doc in cleaned_documents]
    cleaned_documents = [doc.replace("\t", " ") for doc in cleaned_documents]
    cleaned_documents = [re.sub(r'[^A-Za-z0-9 ]+', '', doc) for doc in cleaned_documents]
    return cleaned_documents


def document_tokenize(cleaned_documents):
    stop_words = list(set(stopwords.words('english')))
    document_tokens = [word_tokenize(document) for document in cleaned_documents ]
    tokens=[[token for token in document_tokens if not token in stop_words] for document_tokens in document_tokens ]
    return tokens

def doucments_lemmatizer(documents_tokens):
    tokens=[[WordNetLemmatizer().lemmatize(token) for token in doc_tokens] for doc_tokens in documents_tokens]
    return tokens

def token_frequency_filter(documents_tokens,threshold):
    frequency = defaultdict(int)
    for doc_tokens in documents_tokens:
        for token in doc_tokens:
            frequency[token] += 1
    tokens = [[token for token in doc_tokens if frequency[token] > threshold] for doc_tokens in documents_tokens]
    return tokens


def text_processing(all_documents):
    preprocessed_documents=preprocessing_documents(all_documents)
    documents_tokens=document_tokenize(preprocessed_documents)
    tokens=doucments_lemmatizer(documents_tokens)
    tokens=token_frequency_filter(tokens,5)
    dictionary = Dictionary(tokens)
    corpus = [dictionary.doc2bow(text) for text in tokens]

    return tokens,dictionary,corpus