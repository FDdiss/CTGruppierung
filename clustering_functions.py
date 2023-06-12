import os
import numpy as np
from numpy import save
import pandas as pd
import pickle
import time

import get_greyboxes_path as gr

import scipy
import scipy.misc
from scipy.cluster.hierarchy import dendrogram,linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

import matplotlib.pyplot as plt
from pylab import rcParams

import sklearn
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import sklearn.metrics as sm
from sklearn.preprocessing import scale
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import seaborn as sb

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itk
import itkwidgets
from skimage import filters, exposure

from urllib.request import urlretrieve




#def normalize_features(dataset):


def k_means(pandas_single_file_path,num_clusters=5,number_boxes=False,DMC_list=False,savepath=False,version=False,plot=False):    
    
    if savepath:
        savepath_file=savepath+'/'+os.path.basename(pandas_single_file_path)+'_labels_kmeans_'+str(num_clusters)
        if version:
            savepath_file=savepath_file+'_'+str(version)
    print("check 1")

    #### opendataset
    with open(pandas_single_file_path, "rb") as bb_data:
        pd_data=pd.read_pickle(bb_data)
    print("check 2")

    if DMC_list:
        pd_data=pd_data.loc[pd_data['DMC'] in DMC_list]
    print("check 3")

    if number_boxes:
        pd_data=pd_data.sample(number_boxes)
    print("check 4")

    vector=pd_data.filter(regex="feature")
    vector=vector.loc[:, (vector != 0).any(axis=0)]
    vector=vector.apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=0)

    kmeans=KMeans(n_clusters=num_clusters).fit(pd_data.filter(regex="feature"))
    print("check 5")

    labels=kmeans.labels_
    cluster,number= np.unique(labels,return_counts=True)
    for x in range(0,len(cluster)):
        print('Number auf Points in Cluster %d:' %cluster[x]+(' %d' %number[x]))
    print("check 6")

    if savepath:
        pd_data=pd_data.assign(kmeans_label=labels)
        pd_data.to_pickle(savepath_file)
        print('saved')

    print(pd_data)

    if plot==True:
        plt.rcParams['figure.figsize'] = (18, 8)
        plt.hist(pd_data["kmeans_label"],bins=range(0,num_clusters+1), align="mid", rwidth=0.7)
        plt.title("Histogram eps=10 min_samples=6")
        #plt.ylim(0, )
        #plt.xlim(1)
        plt.show()

        # number=4
        # plotlist=pd_data[pd_data["kmeans_label"]==number].sample(1)
        # a=np.load(gr.get_greyboxes_paths(greyboxes_data_path,sized=False,DMC=plotlist["DMC"].iloc[0],ID=plotlist["Box-ID"].iloc[0])[0])
        # itkwidgets.view(a,rotate=True, axes=True,vmin=4000,vmax=17000,gradient_opacity=0.9)


def agglomerative_cl(pandas_single_file_path,num_clusters=5,number_boxes=False,DMC_list=False,savepath=False,version=False,plot=False,cluster=True,savelinkage=False):
    
    if savepath:
        savepath_file=savepath+'/'+os.path.basename(pandas_single_file_path)+'_labels_Agg_'+str(num_clusters)
        if version:
            savepath_file=savepath_file+'_'+str(version)

    #### open dataset
    with open(pandas_single_file_path, "rb") as bb_data:
        pd_data=pd.read_pickle(bb_data)

    #### randomly select boxes if number_boxes is given
    if number_boxes:
        pd_data=pd_data.sample(number_boxes)
    #### save features into vector
    vector=pd_data.filter(regex="feature")
    vector=vector.loc[:, (vector != 0).any(axis=0)]
    vector=vector.apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=0)
    #Configure the output and matplotlib inline
    np.set_printoptions(precision=4,suppress=True)
    rcParams["figure.figsize"] =100,100
    sb.set_style("whitegrid")


    if plot==True:
        #### calculate distance
        z = linkage(vector,"single") #ward, single, average, complete, hier kann man variieren-> siehe Powerpoint für Unterschiede

        #### linkage abspeichern falls erwünscht
        if savelinkage==True and savepath==True:
            save(savepath_file+"linkage.npy",z)

        #generate dendrogram
        #Dendogramm mit fester Anzahl an Cluster bestimmen
        #Anzahl an Cluster bestimmen
        #number_cluster=30
        #R=dendrogram(z,truncate_mode= "lastp", p=number_cluster, leaf_rotation=45,leaf_font_size=15, show_contracted=True) # truncate_mode= "lastp" führt zu p Cluster

        #Dendogramm mit Anzahl und teilästen bestimmen
        dendrogram(z,truncate_mode= "level", p =4, leaf_rotation=45,leaf_font_size=15, show_contracted=True,color_threshold=50) #truncate_mode= "level" führt dazu, dass immer p Unterteilungen gemacht werden
        plt.title("Truncated Hierachial Clustering Dendrogram")
        plt.xlabel("Cluster Size")
        plt.ylabel("Distance")
        #divide the cluster 
        plt.axhline(y=20)
        plt.axhline(y=15) #Add a horizontal line across the axis.
        plt.axhline(5)
        plt.axhline(10)
        plt.show()

    #Clustering
    if cluster==True:
        print('erstelle Modell')
        HClustering = AgglomerativeClustering(n_clusters=num_clusters , affinity="euclidean",linkage="single") #build the model
        print('starte Clustering')
        aggloclust = HClustering.fit(vector) #fit the model on the dataset
        print('Clustering beendet')
        labels = aggloclust.labels_
        pd_data=pd_data.assign(AkC_label=labels)
        if plot==True:
            plt.rcParams['figure.figsize'] = (18, 8)
            num_clusters=len(pd_data["AkC_label"].unique())
            plt.hist(pd_data["AkC_label"],bins=range(0,num_clusters+1), align="mid", rwidth=0.7)
            plt.title("Histogram eps=10 min_samples=6")
            plt.show()
    if savepath:
        pd_data.to_pickle(savepath_file)
        print('saved')

def dbscann(pandas_single_file_path,eps=5,min_samples=2,number_boxes=False,DMC_list=False,savepath=False,version=False,eps_plot=False,cluster=True,cluster_method="dbscan",plot=False):

    #### opendataset
    with open(pandas_single_file_path, "rb") as bb_data:
        pd_data=pd.read_pickle(bb_data)
    if number_boxes:
        pd_data=pd_data.sample(number_boxes)
    vector=pd_data.filter(regex="feature") #only features
    vector=vector.loc[:, (vector != 0).any(axis=0)] #remove only zero features
    vector=vector.apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=0) #min max skaling

    ##############
    #DB-Scann
    ##############
    #Eps bestimmen

    if eps_plot==True:

        #Calculate the average distance between each point in the data set and its 2 nearest neighbors (my selected MinPts value).
        neighbors = NearestNeighbors(n_neighbors=2) #choose n_neighbors = min_samples
        neighbors_fit = neighbors.fit(vector)
        distances, indices = neighbors_fit.kneighbors(vector)

        #Sort distance values by ascending value and plot
        distances = np.sort(distances, axis=0)
        distances = distances[:,1]
        plt.plot(distances)
        #title: Points sorted by distance to the Xth nearest neighbor
        plt.show()

    #Clustering
    if cluster==True:
        if cluster_method=='dbscan':
            optic_db=OPTICS(max_eps=eps,min_samples=min_samples,cluster_method=cluster_method,metric='manhattan').fit(vector)

        if cluster_method=='xi':
            optic_db=OPTICS(max_eps=eps,min_samples=min_samples,cluster_method=cluster_method,metric='manhattan').fit(vector)
        
        labels = optic_db.labels_
        no_clusters = len(np.unique(labels) )
        no_noise =np.sum(np.array(labels) == -1, axis=0)
        no_labels = (len(labels))
        print('Number of boxes: %d' % no_labels)  
        print('Estimated number of clusters: %d' % no_clusters)
        print('Estimated number of noise points: %d' % no_noise)
        
        cluster,number= np.unique(labels,return_counts=True)
        for x in range(0,len(cluster)):
            print('Number auf Points in Cluster %d:' %cluster[x]+(' %d' %number[x]))

        pd_data=pd_data.assign(DB_label=labels)
        if plot==True:
            plt.rcParams['figure.figsize'] = (18, 8)
            num_clusters=len(pd_data["DB_label"].unique())
            plt.hist(pd_data["DB_label"],bins=range(0,num_clusters+1), align="mid", rwidth=0.7)
            plt.title("Histogram eps=10 min_samples=6")
            #plt.ylim(0, )
            #plt.xlim(1)
            plt.show()

        if savepath:
            savepath_file=savepath+'/'+os.path.basename(pandas_single_file_path)+'_labels_dbscann_'+str(num_clusters)
            if version:
                savepath_file=savepath_file+'_'+str(version)
            if cluster_method=="dbscan":
                pd_data=pd_data.assign(DB_label=labels)
            elif cluster_method=="xi":
                pd_data=pd_data.assign(XI_label=labels)
            pd_data.to_pickle(savepath_file)
            print('saved')

def plot3D_2(imgs, labels, num_Cluster):
    font_size= 10
    palette = np.array(sb.color_palette("hls", 18))
    #my_cmap = ListedColormap(palette)
    my_cmap = sb.light_palette("Navy", as_cmap=True) # construct cmap
    fig = plt.figure(dpi=200, figsize=(32,32))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter3D(imgs[:, 0], imgs[:, 1], imgs[:, 2], c=labels)
    for label in (ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
        label.set_fontsize(font_size)
    #ax.set_xlim(xmin=0.0, xmax=20)
    #plt.xlim(0, 20)
    #plt.ylim(-20, 20)
    #plt.zlim(-20, 20)
    ax.axis('tight')
    fig.colorbar(scatter) 
    return fig, ax, scatter

def plot2D(imgs, labels, num_Cluster):
    font_size= 10
    palette = np.array(sb.color_palette("hls", 18))
    #my_cmap = ListedColormap(palette)
    my_cmap = sb.light_palette("Navy", as_cmap=True) # construct cmap
    fig = plt.figure(dpi=200, figsize=(32,32))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(imgs[:, 0], imgs[:, 1], c=labels, s=1)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(font_size)
    #ax.set_xlim(xmin=0.0, xmax=20)
    #plt.xlim(0, 20)
    #plt.ylim(-20, 20)
    #plt.zlim(-20, 20)
    ax.axis('tight')
    fig.colorbar(scatter) 
    return fig, ax, scatter

def plot_cluster_T_SNE(pandas_single_file_path,label_name='NA',savepath=False,version=False,number_boxes=False):

    if savepath:
        savepath_file=savepath+'/'+os.path.basename(pandas_single_file_path)+'_TSNE'
        if version:
            savepath_file=savepath_file+'_'+str(version)

    with open(pandas_single_file_path, "rb") as bb_data:
        pd_data=pd.read_pickle(bb_data)
    if number_boxes:
        pd_data=pd_data.sample(number_boxes)
    if label_name=='NA':
        label_name=pd_data.filter(regex="label").columns[-1]
    num_clusters=len(pd_data[label_name].unique())

    print(f'Anzahl an Grauboxen: {len(pd_data)}')
    print(f'Anzahl an Clustern: {num_clusters}')
    print(f'Spaltenname der Labels: {label_name}')


    vector=pd_data.filter(regex="feature") #only features
    vector=vector.loc[:, (vector != 0).any(axis=0)] #remove only zero features
    vector=vector.apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=0) #min max skaling

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(vector)
    print(pca_result.shape)
    #print(pca_result)
    print(type(pca_result))
    pd_data['PCA_1'] = pca_result[:,0]
    pd_data['PCA_2'] = pca_result[:,1]
    pd_data['PCA_3'] = pca_result[:,2]
    #sns.palplot(np.array(sns.color_palette("hls", num_Cl)))
    Y = np.array(pd_data[label_name])
    fig, ax, scatter = plot3D_2(pca_result, Y, num_clusters)
    plt.savefig('1_pca_3D.png')
    plt.show()
    print(pd_data[[label_name,"PCA_1","PCA_2","PCA_3"]])
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(pd_data[["PCA_1","PCA_2","PCA_3"]])
    print(tsne_results)
    pd_data['2D_TSNE_x'] = tsne_results[:,0]
    pd_data['2D_TSNE_y'] = tsne_results[:,1]
    #new_df['2D_TSNE_z'] = tsne_results[:,2]

    fig, ax, scatter = plot2D(tsne_results, Y, num_clusters)
    plt.savefig('2_tsne_2D.png')
    plt.show()





directory_path = 'D:/AnomalieKI' #working directory
greyboxes_data_path = directory_path+'/data/04_greyboxes_npy' #where the greyboxes are located
pandas_data_path = directory_path+'/Data/06_pandas_dataset_with_features' #where pandas with boundingboxes and features is located
pandas_single_file_path=pandas_data_path+r'/pd_bb_v02_combined_max.s_43_min.s_03_with_16_features_v7r6'# neues
pandas_data_path_labels=directory_path+'/Data/07_pandas_dataset_with_labels'
pandas_single_file_path_labels=pandas_data_path_labels+r'/pd_bb_v02_combined_max.s_43_min.s_03_with_16_features_v7r6_labels_kmeans_300'# neues



savepath=r'D:\AnomalieKI\Data\07_pandas_dataset_with_labels'
#k_means(pandas_single_file_path, num_clusters=300, savepath=savepath,plot=False)
#dbscann(pandas_single_file_path, eps=0.04, savepath=savepath,eps_plot=True,plot=True)
#agglomerative_cl(pandas_single_file_path, num_clusters=300,savepath=savepath,plot=False)

plot_cluster_T_SNE(pandas_single_file_path_labels)