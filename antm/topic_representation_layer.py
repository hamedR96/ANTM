import pandas as pd
from antm.ctfidf import CTFIDFVectorizer
from sklearn.feature_extraction.text import CountVectorizer

def rep_prep(cluster_df):
    clusters_df = pd.concat(cluster_df)

    clusters_df_copy=clusters_df.copy()
    clusters_df_copy.loc[:,"num_doc"]=1
    clusters_df=clusters_df_copy

    documents_per_topic_per_time= clusters_df.groupby(["slice_num","C"], as_index=False).agg({'content': ' '.join,"num_doc":"count"})
    documents_per_topic_per_time=documents_per_topic_per_time.reset_index().rename(columns={"index":"cluster"})

    return documents_per_topic_per_time


def ctf_idf_topics(docs_per_class,words,ctfidf,num_terms):
    topics=[]
    for label in docs_per_class:
        topic=[]
        for index in ctfidf[int(label)].argsort()[:num_terms]:
            topic.append(words[index])
        topics.append(topic)
    return topics

def ctfidf_rp(dictionary,documents_per_topic_per_time,num_doc,num_words=10):
    count_vectorizer= CountVectorizer(vocabulary=dictionary.token2id).fit(documents_per_topic_per_time.content)
    words= count_vectorizer.get_feature_names_out()
    count= count_vectorizer.transform(documents_per_topic_per_time.content)
    ctfidf= CTFIDFVectorizer().fit_transform(count, n_samples=num_doc).toarray()
    topics_representations=ctf_idf_topics(documents_per_topic_per_time.cluster,words,ctfidf,num_words)
    output = documents_per_topic_per_time.assign(topic_representation=topics_representations)
    return output


def topic_evolution(list_tm,output):
    evolving_topics = []
    for et in list_tm:
        evolving_topic = []
        for topic in et:
            cl = int(float(topic.split("-")[1]))
            win = int(float(topic.split("-")[0]))
            t = output[output["slice_num"] == win]
            t = t[t["C"] == cl]
            evolving_topic.append(t.topic_representation.to_list()[0])
        evolving_topics.append(evolving_topic)
    evolving_topics_df = pd.DataFrame({'evolving_topics': evolving_topics})
    return evolving_topics_df
