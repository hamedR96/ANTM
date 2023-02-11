import glob
import os
import pickle
import random

import pandas as pd
from matplotlib import pyplot as plt

from antm.aligned_clustering_layer import aligned_umap, hdbscan_cluster, draw_cluster, clustered_df, plot_alignment, \
    alignment_procedure, dt_creator, clustered_cent_df
from antm.sws import sws
from antm.contextual_embedding_layer import contextual_embedding
from antm.topic_representation_layer import rep_prep, text_processing,ctfidf_rp, topic_evolution

class ANTM:
    def __init__(self, df, overlap, window_length, mode="data2vec", umap_dimension_size=5, umap_n_neighbors=15,
                 partioned_clusttering_size=5, num_words=10, show_2d_plot=False,path=os.getcwd()):
        self.df = df
        self.overlap = overlap
        self.window_length = window_length
        self.mode = mode
        self.umap_dimension_size = umap_dimension_size
        self.umap_n_neighbors = umap_n_neighbors
        self.partioned_clusttering_size = partioned_clusttering_size
        self.num_words = num_words
        self.show_2d_plot = show_2d_plot

        if not path== os.getcwd():
            if not os.path.exists(path): os.mkdir(path)
        self.path = path

        self.df_embedded = None
        self.umap_embeddings_clustering = None
        self.umap_embeddings_visulization = None
        self.clusters=None
        self.slices=None
        self.arg1_umap=None
        self.arg2_umap=None
        self.cluster_df=None
        self.clustered_df_cent=None
        self.clustered_np_cent=None
        self.dt=None
        self.concat_cent=None
        self.df_tm=None
        self.list_tm=None
        self.documents_per_topic_per_time=None
        self.tokens=None
        self.dictionary=None
        self.corpus=None
        self.output=None
        self.evolving_topics=None


    def fit(self,save=True):
        self.df_embedded = contextual_embedding(self.df, mode=self.mode)
        self.slices, self.arg1_umap, self.arg2_umap = sws(self.df_embedded, self.overlap, self.window_length)
        self.umap_embeddings_clustering, self.umap_embeddings_visulization = aligned_umap(
            self.arg1_umap, self.arg2_umap, n_neighbors=self.umap_n_neighbors,
            umap_dimension_size=self.umap_dimension_size)
        self.clusters = hdbscan_cluster(self.umap_embeddings_clustering, self.partioned_clusttering_size)
        if not os.path.exists(self.path+"/results"): os.mkdir(self.path+"/results")
        for i in range(len(self.clusters)):
            draw_cluster(self.clusters[i], self.umap_embeddings_visulization[i], "time_frame_" + str(i),
                         show_2d_plot=self.show_2d_plot,path=self.path)
        self.cluster_df = clustered_df(self.slices, self.clusters)
        self.clustered_df_cent, self.clustered_np_cent = clustered_cent_df(self.cluster_df)
        self.dt, self.concat_cent = dt_creator(self.clustered_df_cent)
        self.df_tm = alignment_procedure(self.dt, self.concat_cent)
        self.list_tm = plot_alignment(self.df_tm, self.umap_embeddings_visulization, self.clusters,self.path)
        self.documents_per_topic_per_time = rep_prep(self.cluster_df)
        self.tokens, self.dictionary, self.corpus = text_processing(self.df.content.values)
        self.output = ctfidf_rp(self.dictionary, self.documents_per_topic_per_time, num_doc=len(self.df), num_words=self.num_words)
        self.evolving_topics=topic_evolution(self.list_tm, self.output)
        if save: self.save()


    def save(self):

        if not os.path.exists(self.path+"/model"): os.mkdir(self.path+"/model")

        self.df_embedded.to_pickle(self.path+"/model/embedding_df")

        with open(self.path+"/model/slices", "wb") as fp:  # Pickling
            pickle.dump(self.slices, fp)

        with open(self.path+"/model/umap_embeddings_clustering", "wb") as fp:  # Pickling
            pickle.dump(self.umap_embeddings_clustering, fp)

        with open(self.path+"/model/umap_embeddings_visulization", "wb") as fp:  # Pickling
            pickle.dump(self.umap_embeddings_visulization, fp)

        with open(self.path+"/model/clusters", "wb") as fp:  # Pickling
            pickle.dump(self.clusters, fp)

        self.df_tm.to_pickle(self.path+"/model/df_tm")

        self.output.to_pickle(self.path + "/model/output")

        self.evolving_topics.to_pickle(self.path + "/model/evolving_topics")


    def load(self):

        self.df_embedded=pd.read_pickle(self.path+"/model/embedding_df")

        with open(self.path + "/model/slices", "rb") as fp:  # Pickling
            self.slices=pickle.load(fp)

        with open(self.path + "/model/umap_embeddings_clustering", "rb") as fp:
            self.umap_embeddings_clustering = pickle.load(fp)

        with open(self.path + "/model/umap_embeddings_visulization", "rb") as fp:
            self.umap_embeddings_visulization = pickle.load(fp)

        with open(self.path + "/model/clusters", "rb") as fp:
            self.clusters = pickle.load(fp)

        self.df_tm=pd.read_pickle(self.path+"/model/df_tm")

        self.list_tm = plot_alignment(self.df_tm, self.umap_embeddings_visulization, self.clusters,self.path)

        for i in range(len(self.clusters)):
            draw_cluster(self.clusters[i], self.umap_embeddings_visulization[i], "time_frame_" + str(i),
                         show_2d_plot=self.show_2d_plot,path=self.path)

        self.cluster_df = clustered_df(self.slices, self.clusters)

        self.clustered_df_cent, self.clustered_np_cent = clustered_cent_df(self.cluster_df)

        self.dt, self.concat_cent = dt_creator(self.clustered_df_cent)

        self.documents_per_topic_per_time = rep_prep(self.cluster_df)

        self.tokens, self.dictionary, self.corpus = text_processing(self.df.content.values)

        self.output=pd.read_pickle(self.path+"/model/output")

        self.evolving_topics = pd.read_pickle(self.path + "/model/evolving_topics")

    def random_evolution_topic(self):
        random_element = random.choice(self.list_tm)
        list_words = []
        for i in range(len(random_element)):
            # print(random_element[i][0])
            cl = int(random_element[i].split("-")[1])
            win = int(random_element[i].split("-")[0])
            t = self.output[self.output["slice_num"] == win]
            t = t[t["C"] == cl]
            list_words.append(list(t.topic_representation)[0])
            # print(list(t.topic_representation))

        plt.figure(figsize=(15, 10))
        for i in range(len(random_element)):
            cl = int(random_element[i].split("-")[1])
            win = int(random_element[i].split("-")[0])
            labels = self.clusters[win - 1].labels_
            data = self.umap_embeddings_visulization[win - 1]
            data = data.assign(C=labels)
            data = data[data["C"] == cl]
            plt.scatter(data[0], data[1], label=list_words[i])
            plt.legend()

        plt.savefig(self.path+'/results/random_topic_evolution.png')
        plt.show()

    def save_evolution_topics_plots(self,display=False):
        for j in range(len(self.list_tm)):
            list_words = []
            random_element = self.list_tm[j]
            for i in range(len(random_element)):
                # print(random_element[i])
                cl = int(random_element[i].split("-")[1])
                win = int(random_element[i].split("-")[0])
                t = self.output[self.output["slice_num"] == win]
                t = t[t["C"] == cl]
                list_words.append(list(t.topic_representation)[0])
                # print(list(t.topic_representation))
            fig = plt.figure(figsize=(15, 10))
            for i in range(len(random_element)):
                cl = int(random_element[i].split("-")[1])
                win = int(random_element[i].split("-")[0])
                labels = self.clusters[win - 1].labels_
                data = self.umap_embeddings_visulization[win - 1]
                data = data.assign(C=labels)
                data = data[data["C"] == cl]
                plt.scatter(data[0], data[1], label=list_words[i])
                plt.legend()
            if not os.path.exists(self.path+"/results/evolving_topics"): os.mkdir(self.path+"/results/evolving_topics")
            plt.savefig(self.path+'/results/evolving_topics/topic_evolution_' + str(j) + '.png')
            if display:
                plt.show()
            plt.close(fig)

    def plot_clusters_over_time(self):

        filenames = glob.glob(self.path+"/results/partioned_clusters/*.png")

        # number of subplots to display
        n_subplots = len(filenames)

        # Plotting figure with 4 rows and 3 columns
        fig, axs = plt.subplots((n_subplots // 4) + 1, 4, figsize=(15, 15))
        axs = axs.ravel()
        ex = len(axs) - n_subplots
        for i in range(ex):
            axs[-i - 1].axis('off')
        # Adding each png file to the figure
        for i, png in enumerate(filenames):
            img = plt.imread(png)
            axs[i].imshow(img)
            axs[i].axis('off')
            axs[i].set_title(f'Time Frame {i + 1}')

        # Adding subtitle to the figure
        plt.suptitle('Dynamic Document Embeddings and Clusters in each Time Frame')
        plt.savefig(self.path + "/results/partioned_topics.png")
        # Showing the figure
        plt.show()

    def plot_evolving_topics(self):

        filenames = glob.glob(self.path+"/results/evolving_topics/*.png")

        # number of subplots to display
        n_subplots = len(filenames)

        # Plotting figure with 3 columns
        fig, axs = plt.subplots((n_subplots // 3) + 1, 3, figsize=(15, 15))
        axs = axs.ravel()
        ex = len(axs) - n_subplots
        for i in range(ex):
            axs[-i - 1].axis('off')
        # Adding each png file to the figure
        for i, png in enumerate(filenames):
            img = plt.imread(png)
            axs[i].imshow(img)
            axs[i].axis('off')
            axs[i].set_title(f'Evolving Topic {i + 1}')

        # Adding subtitle to the figure
        plt.suptitle('Evolution of Evolving Topics')
        # Showing the figure
        plt.savefig(self.path+"/results/evolving_topics.png")
        plt.show()


