import os
#import get_greyboxes_path as gbp
import numpy as np
import pandas as pd
import pickle

# import scipy
# from scipy.cluster.hierarchy import dendrogram,linkage
# from scipy.cluster.hierarchy import fcluster
# from scipy.cluster.hierarchy import cophenet
# from scipy.spatial.distance import pdist

import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sb

import sklearn
from sklearn import datasets
from sklearn.cluster import OPTICS, cluster_optics_dbscan
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

ebs=False #ebs bestimmen
Hole=False #ganzes Pandas?
Cluster=True #Clustern?

eps=6#13#6 aus Graph der neares Neigbors
min_samples=6
cluster_method='dbscan'
save=True #Clustering speichern? Dann auf folgenden Pfad
savepath_file=r'D:\AnomalieKI\test_waldherr\07_pandas_dataset_with_labels\Optics_DMCs_'+DMC[0]+'_'+DMC[1]+'_eps'+str(eps)+'_samples'+str(min_samples)+'_pd_bb_v02_combined_with_16_features_with_'+cluster_method+'_labels_max.s_43_min.s_03_v7_r6'
#savepath_file=r'C:\Users\waldherr\Desktop\umziehen\Optics_DMCs_'+DMC[0]+'_'+DMC[1]+'_eps'+str(eps)+'_samples'+str(min_samples)+'_pd_bb_v02_combined_with_16_features_with_'+cluster_method+'_labels_max.s_43_min.s_03_v7_r6'

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
#DB-Scann
##############
#Eps bestimmen

if ebs:

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

    plt.plot(distances)
    #plt.ylim(bottom=0, top=1000)
    plt.xlim(left=40000, right=41200)
    plt.show()


#Clustering
if Cluster:
    if cluster_method=='dbscan':
        optic_db=OPTICS(max_eps=eps,min_samples=min_samples,cluster_method=cluster_method,metric='manhattan').fit(vector)

    if cluster_method=='xi':
        optic_db=OPTICS(max_eps=eps,min_samples=min_samples,cluster_method=cluster_method,metric='manhattan').fit(vector)
    labels = optic_db.labels_
    no_clusters = len(np.unique(labels) )
    no_noise =np.sum(np.array(labels) == -1, axis=0)
    no_labels = (len(labels))
    print('Estimated no. of labels: %d' % no_labels)  
    print('Estimated no. of clusters: %d' % no_clusters)
    print('Estimated no. of noise points: %d' % no_noise)
    
    cluster,number= np.unique(labels,return_counts=True)
    for x in range(0,len(cluster)):
        print('Number auf Points in Cluster %d:' %cluster[x]+(' %d' %number[x]))
    #print(number)


    if save:
        pandas_bb_and_features_and_labels_reduced=pandas_bb_and_features_reduced.assign(OP_label=labels)
        pandas_bb_and_features_and_labels_reduced.to_pickle(savepath_file)
        print('saved')
#print(pandas_bb_and_features_and_labels_reduced)
