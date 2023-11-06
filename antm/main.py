import glob
import os
import pickle
import random
import torch
import pandas as pd
from matplotlib import pyplot as plt
from antm.text_processing import text_processing
from antm.aligned_clustering_layer import aligned_umap, hdbscan_cluster, draw_cluster, clustered_df, plot_alignment, \
    alignment_procedure, dt_creator, clustered_cent_df
from antm.sws import sws
from antm.contextual_embedding_layer import contextual_embedding
from antm.topic_representation_layer import rep_prep,ctfidf_rp, topic_evolution
from antm.cm import coherence_model
from antm.diversity_metrics import proportion_unique_words,pairwise_jaccard_diversity

class ANTM:
    def __init__(self, df, overlap, window_length, mode="bert", umap_dimension_size=5, umap_n_neighbors=15,
                 embedding_vectors=None, umap_embeddings_clustering=None, umap_embeddings_visulization=None,
                 device=None , partioned_clusttering_size=10, num_words=10, show_2d_plot=False, show_3d_plot=False,path=os.getcwd()):
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


        if device==None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device=device

        self.embedding_vectors = embedding_vectors
        self.umap_embeddings_clustering = umap_embeddings_clustering
        self.umap_embeddings_visulization = umap_embeddings_visulization
        self.df_embedded = None
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
        self.topics=None
        self.slice_num=None
        self.show_3d_plot=show_3d_plot
        self.periodwise_puw_diversity=None
        self.periodwise_pairwise_jaccard_diversity=None
        self.periodwise_topic_coherence=None

    def fit(self, save=True):
        # Contextual embedding
        if self.embedding_vectors is None:
            print("contextual document embedding is initiated...")
            self.df_embedded = contextual_embedding(self.df, mode=self.mode,device=self.device)
        else:
            print("contextual document embedding provided ---> skip")
            self.df["embedded"]=self.embedding_vectors
            self.df_embedded=self.df
        # Sliding window segmentation
        print("Sliding Window Segmentation is initialized...")
        self.slices, self.arg1_umap, self.arg2_umap = sws(self.df_embedded, self.overlap, self.window_length)

        # Aligned dimensionality reduction
        if self.umap_embeddings_clustering is None or self.umap_embeddings_visulization is None:
            print("Aligned Dimension Reduction is initialized...")
            self.umap_embeddings_clustering, self.umap_embeddings_visulization = aligned_umap(
                self.arg1_umap, self.arg2_umap, n_neighbors=self.umap_n_neighbors,
                umap_dimension_size=self.umap_dimension_size)
        else:
            print("umap embeddings provided ---> skip")

        print("Sequential Document-cluster association is initialized...")
        self.clusters = hdbscan_cluster(self.umap_embeddings_clustering,
                                                            self.partioned_clusttering_size)
        if not os.path.exists(self.path + "/results"): os.mkdir(self.path + "/results")
        for i in range(len(self.clusters)):
            draw_cluster(self.clusters[i], self.umap_embeddings_visulization[i], "time_frame_" + str(i),
                         show_2d_plot=self.show_2d_plot, path=self.path)
        self.cluster_df = clustered_df(self.slices, self.clusters)
        self.clustered_df_cent, self.clustered_np_cent = clustered_cent_df(self.cluster_df)
        self.dt, self.concat_cent = dt_creator(self.clustered_df_cent)
        print("Cluster Alignment Procedure is initialized...")
        self.df_tm = alignment_procedure(self.dt, self.concat_cent)
        self.list_tm = plot_alignment(self.df_tm, self.umap_embeddings_visulization, self.clusters, self.path,self.show_3d_plot)
        self.documents_per_topic_per_time = rep_prep(self.cluster_df)
        self.tokens, self.dictionary, self.corpus = text_processing(self.df.content.values)
        print("Topic Representation is initialized...")
        self.output = ctfidf_rp(self.dictionary, self.documents_per_topic_per_time, num_doc=len(self.df),
                                num_words=self.num_words)
        print("Topic Modeling is done")
        self.evolving_topics = topic_evolution(self.list_tm, self.output)
        if save: self.save()
        self.slice_num = set(self.output["slice_num"])
        self.topics = [self.output[self.output["slice_num"] == i].topic_representation.to_list() for i in
                       self.slice_num]
        self.topics = list(filter(None, self.topics))
        return self.topics

    def save(self):
        print("Model is saving...")
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
        print("Model is Loading...")
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

        self.list_tm = plot_alignment(self.df_tm, self.umap_embeddings_visulization, self.clusters,self.path,self.show_3d_plot)

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
        self.slice_num = len(set(self.output["slice_num"]))
        self.topics = [self.output[self.output["slice_num"] == i].topic_representation.to_list() for i in
                    range(1, self.slice_num + 1)]
        return self.topics

    def random_evolution_topic(self):
        random_element = random.choice(self.list_tm)
        list_words = []
        for i in range(len(random_element)):
            # print(random_element[i][0])
            cl = int(float(random_element[i].split("-")[1]))
            win = int(float(random_element[i].split("-")[0]))
            t = self.output[self.output["slice_num"] == win]
            t = t[t["C"] == cl]
            list_words.append(list(t.topic_representation)[0])
            # print(list(t.topic_representation))

        plt.figure(figsize=(15, 10))
        for i in range(len(random_element)):
            cl = int(float(random_element[i].split("-")[1]))
            win = int(float(random_element[i].split("-")[0]))
            labels = self.clusters[win - 1]
            data = self.umap_embeddings_visulization[win - 1]
            data = data.assign(C=labels)
            data = data[data["C"] == cl]
            plt.scatter(data[0], data[1], label=[win,list_words[i]])
            plt.legend()

        plt.savefig(self.path+'/results/random_topic_evolution.png')
        plt.show()

    def save_evolution_topics_plots(self,display=False):
        for j in range(len(self.list_tm)):
            list_words = []
            random_element = self.list_tm[j]
            for i in range(len(random_element)):
                # print(random_element[i])
                cl = int(float(random_element[i].split("-")[1]))
                win = int(float(random_element[i].split("-")[0]))
                t = self.output[self.output["slice_num"] == win]
                t = t[t["C"] == cl]
                list_words.append(list(t.topic_representation)[0])
                # print(list(t.topic_representation))
            fig = plt.figure(figsize=(15, 10))
            for i in range(len(random_element)):
                cl = int(float(random_element[i].split("-")[1]))
                win = int(float(random_element[i].split("-")[0]))
                labels = self.clusters[win - 1]
                data = self.umap_embeddings_visulization[win - 1]
                data = data.assign(C=labels)
                data = data[data["C"] == cl]
                plt.scatter(data[0], data[1], label=[win,list_words[i]])
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

    def get_periodwise_puw_diversity(self):
        self.periodwise_puw_diversity=[proportion_unique_words(period, topk=self.num_words) for period in self.topics]
        return  self.periodwise_puw_diversity

    def get_periodwise_pairwise_jaccard_diversity(self):
        self.periodwise_pairwise_jaccard_diversity=[pairwise_jaccard_diversity(period, topk=self.num_words) for period in self.topics]
        return  self.periodwise_pairwise_jaccard_diversity

    def get_periodwise_topic_coherence(self,model="c_npmi"):
        self.periodwise_topic_coherence=[coherence_model(period,self.tokens,self.dictionary,self.num_words,c_m=model) for period in self.topics]
        return self.periodwise_topic_coherence
