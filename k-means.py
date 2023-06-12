import os
#import get_greyboxes_path as gbp
import numpy as np
import pandas as pd
import pickle

import scipy
from scipy.cluster.hierarchy import dendrogram,linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sb

import sklearn
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
from sklearn.preprocessing import scale

from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt

directory_path = 'D:/AnomalieKI' #working directory
#greyboxes_data_path = directory_path+'/data/04_greyboxes_npy' #where the greyboxes are located
pandas_data_path = directory_path+'/Data/06_pandas_dataset_with_features' #where pandas with boundingboxes and features is located
pandas_single_file_path=pandas_data_path+r'/pd_bb_v02_combined_with_16_features_max.s_43_min.s_03_v7r6'# neues

#### PARAMETERS ####
number_features=16
number_boxes=80000
#Liste mit DMC wenn kein DMC dann leere Liste
DMC=['26628261111219','26794161111219'] #aktuell die ersten beiden
#number_cluster=64

Hole=False #ganzes Bauteil?
Cluster=True #Clustern?

num_cluster=400
cluster_method='k-means'
save=True #Clustering speichern? Dann auf folgenden Pfad
#savepath_file=r'D:\AnomalieKI\test_waldherr\07_pandas_dataset_with_labels\Test2_Optics'+'num'+str(number_boxes)+'_eps'+str(eps)+'_samples'+str(min_samples)+'_pd_bb_v01_combined_with_16_features_with_'+cluster_method+'_labels_max.s_43_min.s_03_v7_r4'
savepath_file=r'D:\AnomalieKI\test_waldherr\07_pandas_dataset_with_labels\DMCs_'+DMC[0]+'_'+DMC[1]+'_cluster'+str(num_cluster)+'_pd_bb_v02_combined_with_16_features_with_'+cluster_method+'_labels_max.s_43_min.s_03_v7_r6'

#### opendataset
with open(pandas_single_file_path, "rb") as bb_data:
    pandas_bb_and_features=pd.read_pickle(bb_data)

#ganzes Pandas?
if Hole:
    number_boxes=pandas_bb_and_features.shape[0]
#Zeilen aus Pandas auslesen
if len(DMC)>0:#Nur Zeilen der gewünschten Bauteile aus gesamten Pandas entnehmen
    pandas_bb_and_features_reduced=pd.DataFrame([],columns=pandas_bb_and_features.columns)
    for x in DMC:
        pandas_bb_and_features_reduced=pandas_bb_and_features_reduced.append(pandas_bb_and_features[pandas_bb_and_features['DMC']==x])
else:# sontgewünschte Anzahl an Zeilen von oben entnemen
    pandas_bb_and_features_reduced=pandas_bb_and_features.iloc[:number_boxes,:]#Pandas bei bedarf

pandas_features=pandas_bb_and_features_reduced.iloc[:,-number_features:]#feature auslesen
#print(np.unique(pandas_bb_and_features_reduced.DMC))
vector=pandas_features.values

##############
#k-means
##############

#Clustering
if Cluster:
    kmeans = KMeans(n_clusters=num_cluster).fit(vector)
    labels=kmeans.labels_
    
    cluster,number= np.unique(labels,return_counts=True)
    for x in range(0,len(cluster)):
        print('Number auf Points in Cluster %d:' %cluster[x]+(' %d' %number[x]))


    if save:
        pandas_bb_and_features_and_labels_reduced=pandas_bb_and_features_reduced.assign(kmeans_label=labels)
        pandas_bb_and_features_and_labels_reduced.to_pickle(savepath_file)
        print('saved')
#print(pandas_bb_and_features_and_labels_reduced)
