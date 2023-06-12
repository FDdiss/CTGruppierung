import autoencoder_functions as asf
import numpy as np
import pandas as pd
import pickle as pickle
import tensorflow as tf


directory_path = 'D:/AnomalieKI' #working directory
greyboxes_data_path = directory_path+'/data/04_greyboxes_npy' #where the greyboxes are located 

#### which pandas dataset? ####
pandas_single_file_path=r'D:\AnomalieKI\Data\03_pandas_dataset\pd_bb_v02_combined_max.s_43_min.s_03'

#### set AE version, is saved into pandas dataset, format v07r04 ####
AEversion="v07r06"
#### which model, give path to trained AE model, activate correct line in Code ####
model=r'D:\AnomalieKI\Autoencoder\version_7\run_6\model_v7_r6.hdf5'

#### which checkpoint, if AE is to be created by checkpoint, give necessery parameters for AE, activate correct line in Code ####
checkpoint=r'D:\AnomalieKI\Autoencoder\version_7\run_6\checkpoints\checkpoint_v7_r6_0.00008.hdf5'
k1=64
k2=32
k3=16
dropout=0
number_features=16 #AE number of features
architecture=1 #AE architecture

#### give result file path for the pandas file with features ####
result_file=r'D:\AnomalieKI\Data\06_pandas_dataset_with_features\pd_bb_v02_combined_max.s_43_min.s_03_with_16_features_v7r6'

#### set the device for the calculation ####
device='1' #GPU number 0,1 or 2 or "CPU" for CPU usage

#### CODE ####

if device=='CPU':
    with tf.device('/cpu:0'): #only use CPU
        #### use following line for AE model ####
        #active_model=tf.keras.models.load_model(model) 

        ##### use following two lines for AE model from checkpoint ####
        active_model=asf.autoencoder_compiler_encoderonly(architecture,k1,k2,k3,number_features,dropout=dropout) 
        active_model.load_weights(checkpoint,by_name=True)   

        #### don't change ####
        asf.features_from_combined_pandas_plus_greyboxes_to_pandas(pandas_single_file_path,greyboxes_data_path,result_file,active_model,device=str(device),AEversion=AEvserion)
elif str(device)=='0' or str(device)=='1' or str(device)=='2': #könnte man auch als if str(device) in ["0","1","2"]
    print("hallo")
    with tf.device('/gpu:'+str(device)):  #use gpu
        #### use following line for AE model ####
        #active_model=tf.keras.models.load_model(model) 

        ##### use following two lines for AE model from checkpoint ####
        active_model=asf.autoencoder_compiler_encoderonly(architecture,k1,k2,k3,number_features,dropout=dropout)
        active_model.load_weights(checkpoint,by_name=True)    

        #### don't change ####
        asf.features_from_combined_pandas_plus_greyboxes_to_pandas(pandas_single_file_path,greyboxes_data_path,result_file,active_model,device=str(device),AEversion=AEversion)
else:
    print('device error')


