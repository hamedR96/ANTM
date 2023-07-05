
from gensim.corpora.dictionary import Dictionary
from gensim.models import CoherenceModel


def coherence_model(topics,tokens,dictionary,tr_num,c_m="c_npmi"):
    cm = CoherenceModel(topics=topics, texts=tokens, dictionary=dictionary,
                                  coherence=c_m, processes=1, topn=tr_num).get_coherence()
    return cm


