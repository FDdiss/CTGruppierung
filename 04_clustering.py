#import tensorflow as tf
#import autoencoder_functions as asf
import os
import get_greyboxes_path as gbp
import numpy as np
import pandas as pd
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
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import sklearn.metrics as sm
from sklearn.preprocessing import scale

from numpy import save
from skimage import filters, exposure

directory_path = 'D:/AnomalieKI' #working directory
Autoencoder_path= directory_path+'/Autoencoder' #where the checkpoints will be created
greyboxes_data_path = directory_path+'/data/04_greyboxes_npy' #where the greyboxes are located
pandas_data_path = directory_path+'/Data/06_pandas_dataset_with_features'

#### PARAMETERS ####
pandas_single_file_path=pandas_data_path+r'/pd_bb_v01_combined_with_16_features_max.s_43_min.s_03_v7_r4'
number_features=16
number_greyboxes=5000
savelinkage=True
savepath_file=r'D:\AnomalieKI\Data\07_clustering_data\linkage_pd_bb_v01_combined_with_16_features_max.s_43_min.s_03_v7_r4'

#### CODE ####

#### opendataset
with open(pandas_single_file_path, "rb") as bb_data:
    boxes_total=pd.read_pickle(bb_data)

#### save features (number_features) into vector
vector=boxes_total.iloc[:number_greyboxes,-number_features:]

####print vector
print(vector)

#Configure the output
np.set_printoptions(precision=4,suppress=True)

#matplotlib inline
rcParams["figure.figsize"] =50,50
sb.set_style("whitegrid")

#### calculate distance
z = linkage(vector,"ward") #ward, single, average, complete, hier kann man variieren-> siehe Powerpoint für Unterschiede
if savelinkage==True:
    save(savepath_file,z)

#generate dendrogram
#dendrogram(z,truncate_mode= "lastp", p =12, leaf_rotation=45,leaf_font_size=15, show_contracted=True)
dendrogram(z,leaf_rotation=45,leaf_font_size=15, show_contracted=True)
plt.title("Truncated Hierachial Clustering Dendrogram")
plt.xlabel("Cluster Size")
plt.ylabel("Distance")

#divide the cluster 
plt.axhline(y=15) #Add a horizontal line across the axis.
plt.axhline(5)
plt.axhline(10)
plt.show()




''' Preprocessing
DMC="26628261111219"
array=asf.prepare_array_path_list_for_predict(gbp.get_greyboxes_paths(greyboxes_data_path,DMC=DMC)[0:1])

thresh_otsu = filters.threshold_otsu(array)
thresh_img_otsu_mean = np.where(array < thresh_otsu,0.0,1.0)
print(thresh_otsu) #0.548828125
#save(r'D:\KIAnomalie HiWis\Input_Daten_Autoencoder\Input_ohne_Hintergrund/D123_cube_segmented', thresh_img_otsu_mean)
'''
''' feature export (use feature export functions of asf instead)
k1=64
k2=32
k3=16
dropout=0
number_features=16
architecture=1
device='CPU'
checkpoint=r'D:\AnomalieKI\Autoencoder\version_7\run_2\checkpoints\checkpoint_v7_r2_0.00012.hdf5'
#result_file=r'pandas_bb_combined_with_16_features_max.s_43_min.s_03_v7_r2'
with tf.device('/cpu:0'):
    active_model=asf.autoencoder_compiler_encoderonly(architecture,k1,k2,k3,number_features,dropout=dropout)
    active_model.load_weights(checkpoint,by_name=True)

with open(pandas_single_file_path, "rb") as bb_data:
    boxes_total=pd.read_pickle(bb_data)[:100]

DMC_list=boxes_total["CT-Nummer"].unique()

k=0
for i in DMC_list:
    print(str(k+1),"-ter DMC")
    print(i)
    greyboxes_path_list=gbp.get_greyboxes_paths(greyboxes_data_path,DMC=i)[:100]
    if k==0:
        features=asf.predict_array_path_list_with_active_model(greyboxes_path_list,active_model,device=device)
    else:
        features=np.append(features,asf.predict_array_path_list_with_active_model(greyboxes_path_list,active_model,device=device),axis=0)
    k=k+1

number_features=features.shape[1]
column_list=[]
for i in range(1,number_features+1):
    column_list.append("feature"+f"{i:02d}")

df=pd.DataFrame(features,columns=column_list)
dataset=pd.concat([boxes_total,df],axis=1)
'''
