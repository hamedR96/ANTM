from models.aligned_clustering_layer import aligned_umap, hdbscan_cluster, draw_cluster, clustered_df, plot_alignment, \
    alignment_procedure, dt_creator, clustered_cent_df
from models.sws import sws
from models.contextual_embedding_layer import contextual_embedding
from models.topic_representation_layer import rep_prep, text_processing,ctfidf_rp, topic_evolution


def ANTM(df,overlap,window_length,mode="data2vec",umap_dimension_size=5,umap_n_neighbors=20,partioned_clusttering_size=5,num_words=5):
    df_embedded=contextual_embedding(df,mode=mode)
    slices,arg1_umap,arg2_umap=sws(df_embedded,overlap,window_length)
    umap_embeddings_clustering,umap_embeddings_visulization=aligned_umap(arg1_umap,arg2_umap,n_neighbors=umap_n_neighbors,umap_dimension_size=umap_dimension_size)
    clusters = hdbscan_cluster(umap_embeddings_clustering, partioned_clusttering_size)
    for i in range(len(clusters)):
        draw_cluster(clusters[i],umap_embeddings_visulization[i],"time_frame_"+str(i))
    cluster_df=clustered_df(slices,clusters)
    clustered_df_cent,clustered_np_cent=clustered_cent_df(cluster_df)
    dt,concat_cent=dt_creator(clustered_df_cent)
    df_tm=alignment_procedure(dt,concat_cent)
    list_tm=plot_alignment(df_tm,umap_embeddings_visulization,clusters)
    documents_per_topic_per_time=rep_prep(cluster_df)
    tokens,dictionary,corpus=text_processing(df.abstract.values)
    output=ctfidf_rp(dictionary,documents_per_topic_per_time,num_doc=len(df),num_words=num_words)
    return topic_evolution(list_tm,output)
