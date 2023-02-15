import os
import umap
import hdbscan
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np


def aligned_umap(arg1_umap,arg2_umap,n_neighbors=15,umap_dimension_size=5):
    model_umap_clustering = umap.aligned_umap.AlignedUMAP(
    metric="cosine",
    n_neighbors=n_neighbors,
    alignment_regularisation=0.1,
    alignment_window_size=umap_dimension_size,
    n_epochs=200,
    random_state=42,
    ).fit(arg1_umap, relations = arg2_umap)

    umap_embeddings_clustering=[]
    for j in model_umap_clustering.embeddings_:
        umap_embeddings_clustering.append(pd.DataFrame(j))


    model_umap_visualization = umap.aligned_umap.AlignedUMAP(
    metric="cosine",
    n_neighbors=n_neighbors,
    alignment_regularisation=0.1,
    alignment_window_size=2,
    n_epochs=200,
    random_state=42,
    ).fit(arg1_umap, relations = arg2_umap)
    umap_embeddings_visulization=[]
    for j in model_umap_visualization.embeddings_:
        umap_embeddings_visulization.append(pd.DataFrame(j))

    return umap_embeddings_clustering,umap_embeddings_visulization


def hdbscan_cluster(embedding, size) :
    clusters_labels = []
    c= hdbscan.HDBSCAN(min_cluster_size=size, metric = "euclidean",cluster_selection_method = "eom")
    for e in range(len(embedding)) :
        c.fit(embedding[e])
        clusters_labels.append(c.labels_)
    return clusters_labels

def draw_cluster(cluster_labels,umap,name,show_2d_plot,path):
    labels = cluster_labels
    data=umap
    data = data.assign(C=labels)
    data=data[data["C"]>-1]
    fig = plt.figure(figsize=(15, 10))
    plt.scatter(data[0], data[1], c=data["C"])
    if not os.path.exists(path+"/results/partioned_clusters"): os.mkdir(path+"/results/partioned_clusters")
    plt.savefig(path+"/results/partioned_clusters/"+name+'.png')
    if not show_2d_plot:
        plt.close(fig)
    plt.show()

def clustered_df(slices,clusters_labels):
    clustered_df=[]
    for i in range(len(slices)):
        slice=slices[i]
        labels=clusters_labels[i]
        slice = slice.assign(C= labels)
        slice=slice[slice["C"]>-1]
        slice=slice.reset_index(drop=True)
        clustered_df.append(slice)
    return clustered_df


def clustered_cent_df(clustered_df):
    clustered_df_cent=[]
    clustered_np_cent=[]
    for i in clustered_df:
        de=i[["C","embedding"]]
        de = de.groupby("C")["embedding"].apply(list).reset_index()
        de["embedding_mean"] = de["embedding"].apply(lambda x: np.mean(x, axis=0))
        de=pd.DataFrame(list(de['embedding_mean']))
        clustered_df_cent.append(de)
        clustered_np_cent.append(de.to_numpy())
    return clustered_df_cent,clustered_np_cent


def dt_creator(clustered_df_cent):
    topics_cent=[]
    for i in range(len(clustered_df_cent)):
        t=clustered_df_cent[i].copy().reset_index().rename(columns={"index":"cluster_num"})
        t["window_num"]=i+1
        topics_cent.append(t)
    dt=pd.concat(topics_cent).reset_index(drop=True)
    concat_cent=pd.concat(clustered_df_cent).reset_index(drop=True)
    return dt,concat_cent


def alignment_procedure(dt,concat_cent,umap_n_neighbor=2,umap_n_components=5,min_cluster_size=2):
    umap_args = {'n_neighbors': umap_n_neighbor,
                         'n_components': umap_n_components,
                         'metric': 'cosine'}

    hdbscan_args = {'min_cluster_size': min_cluster_size,
                    'metric': 'euclidean',
                    'cluster_selection_method': 'eom'}

    umap_cent = umap.UMAP(**umap_args).fit(concat_cent)
    cluster_cent = hdbscan.HDBSCAN(**hdbscan_args).fit(umap_cent.embedding_)
    return dt.assign(C=cluster_cent.labels_)



def plot_alignment(df_tm,umap_embeddings_visualization,clusters_labels,path):
    tm = df_tm[["window_num", "cluster_num", "C"]]
    tm["name"] = tm.apply(lambda row: str(row["window_num"]) + "-" + str(row["cluster_num"]), axis=1)
    tm = tm[tm["C"] != -1]
    tm = tm.groupby("C")["name"].apply(list).reset_index()
    list_tm = list(tm["name"])
    #with open("./results/all_topic_evolution.txt", "w") as output:
        #for et in list_tm:
            #output.write("\n")
            #for topic in et:
                #output.write(str(topic) + "\s")

    ccs_list=[]
    for i in range(len(list_tm)):
        #print(i)
        cc_list=[]
        for j in list_tm[i]:

            cl=int(j.split("-")[1])
            win=int(j.split("-")[0])


            labels = clusters_labels[win-1]
            data=umap_embeddings_visualization[win-1]
            data = data.assign(C=labels)
            data=data[data["C"]==cl]
            data["win"]=win
            cc_list.append(data)
        cc_df=pd.concat(cc_list)
        cc_df["evolving_topic"]=i
        ccs_list.append(cc_df)
    ccs_df=pd.concat(ccs_list)

    fig = px.scatter_3d(x=ccs_df[0], y=ccs_df[1], z=ccs_df["win"],
                        color=ccs_df["evolving_topic"],color_continuous_scale=px.colors.sequential.Viridis)
    fig.update_layout(width=1000, height=1000)

    fig.show()
    fig.write_image(path+"/results/fig_3D.png")
    return(list_tm)
