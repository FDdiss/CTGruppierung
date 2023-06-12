import opening_closing_functions as ocf
import scv_reader_functions as srf
import os
import pandas as pd
import pickle as pickle

''' All functions to preprocess CT-Scans into .npy files, pandas files and greyboxes'''

directory_path = 'D:/AnomalieKI' #working directory
#raw_data_path = directory_path+'/Data/00_raw_data_scv' #directory with scv files on D:
raw_data_path= 'E:' #externe Festplette mit scvs
meta_data_path = directory_path+'/Data/01_meta_data_pickle' #directory with meta data
#meta_data_path = directory_path+'/test' #test
raw_array_data_path = directory_path+'/Data/02_raw_data_npy' #directory with complete .npy arrays
pandas_data_path = directory_path+'/Data/03_pandas_dataset' #directory with the bounding boxes pandas files
#pandas_data_path = directory_path+'/test'  #test
OC_greyboxes_data_path = directory_path+'/Data/04_greyboxes_npy' #directory with the greyboxes, each DMC gets sub directory automatically
#OC_greyboxes_data_path = directory_path+'/test' #test

##### if single data for function is needed, set path to file here #####
pandas_single_data_path = pandas_data_path+'/pd_bb_v02_combined_max.s_43_min.s_03'
#raw_array_single_data_path = raw_array_data_path+'/CT_26628261111219.npy'

##### compare two scv files #####
#filepath1=r'D:\AnomalieKI\Data\00_raw_data_scv\26824781111219864785406_2019_12_11_10_47_32 2019-12-11 10_46_50\2GeneralEval\volume.uint16_scv'
#filepath2=r'D:\AnomalieKI\Data\00_raw_data_scv\26824781111219864785406_2019_12_11_10_47_32 2019-12-11 10_46_50\2Join\volume.uint16_scv'
# print(srf.compare_two_scv_files(filepath1,filepath2,header_bytes=1024,output=True))

##### read all scv in raw_data_path and create .npy array file, 2 min for each scv data #####
#a="26134390111219863229508"
#print(len("26134390111219863229508"))
#print(a[-24:-9])
#srf.scv_reader(raw_data_path,raw_array_data_path,meta_data_path,overwrite=False,output=True,delete_background=False)

##### create bounding boxes, 6h for each array, #path version, checks for already created bb with same struc sizes #####
ocf.create_OC_bb_for_all(raw_array_data_path,pandas_data_path,maxstruc=43,minstruc=3,overwrite=False,output=True,gpu=2,OCversion="v02",endstring="x.npy")

#single array version
#ocf.create_OC_boundingboxes(raw_array_single_data_path,pandas_data_path, maxstruc=3, output=True,minstruc=3,gpu=1,OCversion="v02")

###### combine pandas files to one combined pandas file, remove duplicates if found #####
#ocf.combine_pandas_files(pandas_data_path)

##### create greyboxes, resized or not #####
#ocf.create_OC_greyboxes_for_all(raw_array_data_path,pandas_single_data_path,OC_greyboxes_data_path,resize_size=64,save_separate=True,gpu=1)


##### create bounding boxes and grey boxes #####
#do not use anymore
#ocf.create_OC_greyboxes(raw_array_single_data_path,pandas_data_path,OC_greyboxes_data_path,resize_size=64,save_separate=True,struc=7,output=True)