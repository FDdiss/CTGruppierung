import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv3D, Dense, Embedding, MaxPooling3D, AveragePooling3D, Conv3DTranspose, UpSampling3D, SpatialDropout3D, BatchNormalization, Flatten, Dropout
from tensorflow.keras.layers import Input 
from tensorflow.keras import Sequential #
from numpy import load
#from tensorflow.keras.layers import 
from tensorflow.keras import Model #
from tensorflow.keras.optimizers import Adam #
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras import backend as K
import os
from datetime import datetime
import random
from operator import itemgetter
import pandas as pd
import pickle as pickle
import matplotlib.pyplot as plt
import math
import get_greyboxes_path as gbp

'''COMPILER functions'''

### this function defines the architecture. add new architecture int(x) with a new elif ver==x:
def autoencoder_compiler(k1,k2,k3,number_of_features,dropout=0,ver=0,verbosity=False):
    #k1 to k3 are int(the kernel numbers) for the three conv layers in encoder and decoder
    #dropout is to set the float(dropout portion) in the dense layers
    #verbosity prints more infos if True
    
    #VERSION 0 architecture
    if ver==0:
        learning_rate = 0.0001 #fixed learning rate
        if verbosity==1: #print infos. if output=True, then verbose is set to 1 by main function. 1 has to be changed to True for this function
            verbosity=True
        else:
            verbosity=False
        input_data = Input(shape=(64,64,64,1)) #defines the input shape

        # 1.Convolutional Layer:
        x = Conv3D(k1, kernel_size=(10, 10, 10), padding='same', activation='relu')(input_data) 
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)
        x = BatchNormalization()(x)

        # 2.Convolutional Layer:
        x = Conv3D(k2, kernel_size=(5, 5, 5),padding='valid', activation='relu')(x)
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)
        x = BatchNormalization()(x)

        # 3.Convolutional Layer:
        x = Conv3D(k3, kernel_size=(3, 3, 3),padding='valid', activation='relu')(x)
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        
        # Clustering Layer:
        x = Dense(number_of_features, activation='relu')(x) #This is the feature layer

        x = Dense(13824, activation='relu')(x)
        x = tf.reshape(x,(-1,6,6,6,64))

        # 1.Deconvolutional Schicht
        x = Conv3DTranspose(k3, kernel_size=(5, 5, 5), activation='relu')(x) 
        x = UpSampling3D(size=(2, 2, 2))(x)

        # 2.Deconvolutional Schicht
        x = Conv3DTranspose(k2, kernel_size=(5,5,5), activation='relu')(x)
        
        # 3.Deconvolutional Schicht
        x = Conv3DTranspose(k1, kernel_size=(9,9,9), activation='relu')(x)
        x = UpSampling3D(size=(2, 2, 2))(x)

        x = Conv3DTranspose(1, kernel_size=(1, 1, 1), activation='sigmoid')(x) 

        autoencoder = Model(input_data, x, name="3D_Convolutional_AE")

        autoencoder.compile(loss='mean_squared_error',
                optimizer=Adam(lr=learning_rate))  # evt. optimizer 'adadelta'
        return autoencoder
    
    ##VERSION 1 architecture
    elif ver==1:
        if verbosity==1: #print infos. if output=True, then verbose is set to 1 by main function. 1 has to be changed to True for this function
            verbosity=True
        else:
            verbosity=False
        input_data = Input(shape=(64,64,64,1))

        # 1.Convolutional Layer:
        x = Conv3D(k1, kernel_size=(8, 8, 8), padding='same', activation='relu')(input_data) # Hier sigmoid und relu variieren
        x = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x)
        x = BatchNormalization()(x)

        # 2.Convolutional Layer:
        x = Conv3D(k2, kernel_size=(8, 8, 8),padding='same', activation='relu')(x)
        x = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x)
        x = BatchNormalization()(x)


        # 3.Convolutional Layer:
        x = Conv3D(k3, kernel_size=(8, 8, 8),padding='same', activation='relu')(x)
        x = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x)
        x = BatchNormalization()(x)

        x = Flatten()(x)

        #1.Dense Layer encoder:
        if not dropout==0:
            x = Dropout(dropout)(x)
        x = Dense(512, activation='relu')(x) 

        # Clustering Layer:
        x = Dense(number_of_features, activation='relu')(x) #This is the feature layer

        # 1.Dense Layer decoder:
        x = Dense(512, activation='relu')(x) 

        if not dropout==0:
            x = Dropout(dropout)(x)
        x = Dense(8192, activation='relu')(x)
        x = tf.reshape(x,(-1,8,8,8,16))
        
        # 1.Deconvolutional Layer
        x = Conv3DTranspose(k3, kernel_size=(8, 8, 8),padding='same', activation='relu')(x) 
        x = UpSampling3D(size=(2, 2, 2))(x)

        # 2.Deconvolutional Layer
        x = Conv3DTranspose(k2, kernel_size=(8, 8, 8),padding='same', activation='relu')(x)
        x = UpSampling3D(size=(2, 2, 2))(x)
        
        # 3.Deconvolutional Layer
        x = Conv3DTranspose(k1, kernel_size=(8, 8, 8),padding='same', activation='relu')(x)
        x = UpSampling3D(size=(2, 2, 2))(x)

        x = Conv3DTranspose(1, kernel_size=(1, 1, 1), activation='sigmoid')(x) 

        autoencoder = Model(input_data, x, name="3D_Convolutional_AE")

        autoencoder.compile(loss='mean_squared_error',
                optimizer=Adam())  # evt. optimizer 'adadelta'   
        return autoencoder

'''COMPILER functions for only encoding to get feature vector'''

def autoencoder_compiler_encoderonly(ver,k1,k2,k3,number_of_features,dropout=0,verbose=0):
    #VERSION 0 architecture
    if ver==0:
        if verbose==1: #print infos. if output=True, then verbose is set to 1 by main function. 1 has to be changed to True for this function
            verbosity=True
        input_data = Input(shape=(64,64,64,1)) #defines the input shape

        # 1.Convolutional Layer:
        x = Conv3D(k1, kernel_size=(10, 10, 10), padding='same', activation='relu')(input_data) 
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)
        x = BatchNormalization()(x)

        # 2.Convolutional Layer:
        x = Conv3D(k2, kernel_size=(5, 5, 5),padding='valid', activation='relu')(x)
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)
        x = BatchNormalization()(x)

        # 3.Convolutional Layer:
        x = Conv3D(k3, kernel_size=(3, 3, 3),padding='valid', activation='relu')(x)
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        
        # Clustering Layer:
        x = Dense(number_of_features, activation='relu')(x) #This is the feature layer

        autoencoder = Model(input_data, x, name="3D_Convolutional_AE_encoder")

        #autoencoder.compile(loss='mean_squared_error',
        #        optimizer=Adam(lr=learning_rate))  # evt. optimizer 'adadelta'
        return autoencoder
    ##VERSION 1 architecture
    elif ver==1:
        if verbose==1: #print infos. if output=True, then verbose is set to 1 by main function. 1 has to be changed to True for this function
            verbosity=True
        input_data = Input(shape=(64,64,64,1))

        # 1.Convolutional Layer:
        x = Conv3D(k1, kernel_size=(8, 8, 8), padding='same', activation='relu')(input_data) # Hier sigmoid und relu variieren
        x = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x)
        x = BatchNormalization()(x)

        # 2.Convolutional Layer:
        x = Conv3D(k2, kernel_size=(8, 8, 8),padding='same', activation='relu')(x)
        x = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x)
        x = BatchNormalization()(x)


        # 3.Convolutional Layer:
        x = Conv3D(k3, kernel_size=(8, 8, 8),padding='same', activation='relu')(x)
        x = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x)
        x = BatchNormalization()(x)

        x = Flatten()(x)

        #1.Dense Layer encoder:
        if not dropout==0:
            x = Dropout(dropout)(x)
        x = Dense(512, activation='relu')(x) 

        # Clustering Layer:
        x = Dense(number_of_features, activation='relu')(x) #This is the feature layer

        autoencoder = Model(input_data, x, name="3D_Convolutional_AE")

        return autoencoder

'''helper functions'''
'''
def get_greyboxes_paths(directory,sized=True,size=64,single=True,DMC=False,OC_version=False,ID=False): 
    
    # USE gbp.getgreyboxes_path
    # Function got its own .py due to tensorflow import in this one
    # NEW VERSION FOR NEW GREBOXES NAMES
    
    # creates a list of .npy files with specific elements
    #takes a directory path

    #greyboxes are created with a specific string: 
    # greybox_v01_size64_26628261111219_00000.npy
    # greybox_v01_unsized_26628261111219_00000.npy
    # greyboxes_v01_size64_26628261111219.npy
    # greyboxes_v01_unsized_26628261111219.npy 
    #number is only added if it is a file with a single array

    #set "size=int" to get only files with resized arrays of this specific size
    #set "size=False" to get only files with non resized original arrays, overrides size
    #set "single=True" to get only files with a single array
    #set "DMC=26628261111219" to get only files of this DMC Scan, can be int or str
    #set OC_version to get only files of a specific OC_run
    #set ID to only get boxes with this specific ID
    #       allows "int" which will be automatically transformed to a five digit str (00001)
    #       allows "str" which NEED to be five digit
    #       allows "list" of only int or only str of the above formats
    #set "single=False" to get only files with multiple arrays. DEPRECIATED since AE func only takes single array files at the moment.

    IDlist=False
    if not ID==False:
        if type(ID)==str:
            IDstring=ID
        elif type(ID)==int:
            IDstring=f"{ID:05d}"
        elif type(ID)==list:
            IDlist=ID

            if type(IDlist[0])==int:
                IDlist=['{:05d}'.format(item) for item in IDlist]
        else:
            print("EROOR: wrong ID Input")
            IDstring=False

    endstring=".npy"
    sizestring="unsized"
    ID_startpos=-9
    ID_endpos=-4
    DMC_startpos=-18
    DMC_endpos=-4
    size_startpos=-26
    size_endpos=-19
    
    if OC_version==False:
        OC_version_len=0
    else:
        OC_version_len=len(OC_version)
    OC_version_startpos=size_startpos-OC_version_len-1
    OC_version_endpos=size_startpos-1

    if single==True:
        DMC_startpos=DMC_startpos-6
        DMC_endpos=DMC_endpos-6
        size_startpos=size_startpos-6
        size_endpos=size_endpos-6    
        OC_version_startpos=OC_version_startpos-6
        OC_version_endpos=OC_version_endpos-6
    
    if sized==True:
        size_startpos=size_startpos+1
        OC_version_startpos=OC_version_startpos+1
        OC_version_endpos=OC_version_endpos+1
        sizestring="size"+str(size)
    name="greybox_v01_unsized_26628261111219_00000.npy"
    if IDlist==False:
        greyboxes_path_list = [os.path.join(root, name)
                    for root, dirs, files in os.walk(directory)
                    for name in files
                    if (name.endswith(endstring) and name[size_startpos:size_endpos]==sizestring)
                    if (OC_version==False or name[OC_version_startpos:OC_version_endpos]==OC_version)
                    if (DMC==False or name[DMC_startpos:DMC_endpos]==str(DMC))
                    if (ID==False or name[ID_startpos:ID_endpos]==IDstring)]
    else:
        greyboxes_path_list = [os.path.join(root, name)
                    for root, dirs, files in os.walk(directory)
                    for name in files
                    if (name.endswith(endstring) and name[size_startpos:size_endpos]==sizestring)
                    if (OC_version==False or name[OC_version_startpos:OC_version_endpos]==OC_version)
                    if (DMC==False or name[DMC_startpos:DMC_endpos]==str(DMC))
                    if (ID==False or (name[ID_startpos:ID_endpos] in IDlist))]       
    
    return greyboxes_path_list
'''

''' PLOTTING FUNCTIONS '''

def plot_loss_func(array_loss,array_val_loss='NA'): 
    #use to plot loss function

    #plots one or two losses in one graph
    #losses need to be arrays (only tested for arrays)
    #first given array is interpreted as loss
    #second given array is interpreted as validation loss
    plt.plot(array_loss)
    if not (array_val_loss=='NA'): #if second array is given, plot this one too
        plt.plot(array_val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    if array_val_loss=='NA': #if second array is given, print legend for this one too
        plt.legend(['train'], loc='upper left')
    else:
        plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
def plot_sample_64(decoded_array,original_array,slice1=0,slice2=30,slice3=63):
    #print three slices of two given array

    #takes two arrays
    #use for plotting one original array and one decoded array
    #set slices if you want to see specific slices

    fig, axs = plt.subplots(2,3, figsize = (20,10))

    axs[0,0].imshow(decoded_array[:,:,slice1], cmap='gray')
    axs[0,0].set_title('slice:'+str(slice1))

    axs[0,1].imshow(decoded_array[:,:,slice2], cmap='gray')
    axs[0,1].set_title('slice:'+str(slice2))
 
    axs[0,2].imshow(decoded_array[:,:,slice3], cmap='gray')
    axs[0,2].set_title('slice:'+str(slice3))

    axs[1,0].imshow(original_array[:,:,slice1], cmap='gray')
    axs[1,0].set_title('slice:'+str(slice1))

    axs[1,1].imshow(original_array[:,:,slice2], cmap='gray')
    axs[1,1].set_title('slice:'+str(slice2))

    axs[1,2].imshow(original_array[:,:,slice3], cmap='gray')
    axs[1,2].set_title('slice:'+str(slice3))

    plt.show()

def plotsamples_with_checkpoint(data_directory,restart_file,k1,k2,k3,number_of_features,architecture=0,dropout=0,number_plots=3,
                                Boxsized=True,Boxsize=64,Boxsingle=True,BoxDMC=False,BoxOC_version=False,BoxID=False): 
    #takes random samples of greyboxes in directory and plots them in form of three original/decoded slices
    #use when you dont have a saved model yet, but model is still trained

    #data_directory is where the greyboxes are located
    #restart_file is a checkpoint file for the given architecture
    #k1 to k3 are the kernely numbers for the architecture
    #architecture is the architecture number defined in AE_compile function
    #dropout should be 0 for plotting, but can be chosen to any float<=1
    #number_plots is the number of samples chosen and plotted
    #uses only CPU so GPU training is not influenced
    
    with tf.device('/cpu:0'): #only use CPU
        convae = autoencoder_compiler(k1,k2,k3,number_of_features,dropout=dropout,ver=architecture) #build autoencoder model
        convae.load_weights(restart_file) # load weights if wanted
        train_list=gbp.get_greyboxes_paths(data_directory,sized=Boxsized,size=Boxsize,single=Boxsingle,DMC=BoxDMC,OC_version=BoxOC_version,ID=BoxID) #get list of greybox files in given directory
        len_train=len(train_list) #get length of file list
        Array=np.zeros((number_plots,64,64,64)) #build array of data
        k=0 #integer for defining position in array
        for i in create_random_samples_from_data_list(train_list,len_train,number_plots): #creates sample list of length number_plots and save into data array
            Array[k,:,:,:]=np.load(i)/65535 #load and normalize uint16 data
            k=k+1 #next position in array
        decoded_data=convae.predict(Array) #use AE model to create decoded array
        for i in range(0,number_plots): #plot number_plots plots
            plot_sample_64(decoded_data[i,:,:,:],Array[i,:,:,:]) #decoded in the upper row, original in lower row

def plotsamples_with_model(data_directory,model,number_plots=3,
                            Boxsized=True,Boxsize=64,Boxsingle=True,BoxDMC=False,BoxOC_version=False,BoxID=False): 
    #takes random samples of greyboxes in directory and plots them in form of three original/decoded slices
    #use when you dont have a saved model yet, but model is still trained

    #data_directory is where the greyboxes are located
    #restart_file is a checkpoint file for the given architecture
    #k1 to k3 are the kernely numbers for the architecture
    #architecture is the architecture number defined in AE_compile function
    #dropout should be 0 for plotting, but can be chosen to any float<=1
    #number_plots is the number of samples chosen and plotted
    #uses only CPU so GPU training is not influenced
    
    with tf.device('/cpu:0'): #only use CPU
        convae=tf.keras.models.load_model(model) #build autoencoder model
        convae.load_weights(restart_file) # load weights if wanted
        train_list=gbp.get_greyboxes_paths(data_directory,sized=Boxsized,size=Boxsize,single=Boxsingle,DMC=BoxDMC,OC_version=BoxOC_version,ID=BoxID) #get list of greybox files in given directory
        len_train=len(train_list) #get length of file list
        Array=np.zeros((number_plots,64,64,64)) #build array of data
        k=0 #integer for defining position in array
        for i in create_random_samples_from_data_list(train_list,len_train,number_plots): #creates sample list of length number_plots and save into data array
            Array[k,:,:,:]=np.load(i)/65535 #load and normalize uint16 data
            k=k+1 #next position in array
        decoded_data=convae.predict(Array) #use AE model to create decoded array
        for i in range(0,number_plots): #plot number_plots plots
            plot_sample_64(decoded_data[i,:,:,:],Array[i,:,:,:]) #decoded in the upper row, original in lower row

'''FEATURE CREATION FUNCTIONS'''

def prepare_single_array_for_predict(array3D):
    #changes shape of single array to feed into model

    array=np.zeros((1,64,64,64))
    array[0,:,:,:]=array3D
    return array

def prepare_array_path_list_for_predict(array_path_list):
    #loads the single arrays and saves them into one array, normalizes it
    
    list_len=len(array_path_list)
    array=np.zeros((list_len,64,64,64))
    k=0
    for i in array_path_list:
        array[k,:,:,:]=np.load(i)/65535
        k=k+1
    return array

def features_of_single_array_with_checkpoint(array3D,checkpoint,architecture=1,k1=16,k2=32,k3=64,number_of_features=16,dropout=0,device='CPU'):
    #takes an NORMALIZED (64,64,64) array and returns its feature vektor as array
    #compiles a new model! Don't use in a for-statement for multiple arrays
    #use predict_single_array_with_active_model

    array=prepare_single_array_for_predict(array3D)
    if device=='CPU':
        with tf.device('/cpu:0'): #only use CPU
            convae = autoencoder_compiler_encoderonly(architecture,k1,k2,k3,number_of_features,dropout=dropout) #build autoencoder model
            convae.load_weights(checkpoint,by_name=True) # load weights if wanted
            decoded_array=convae.predict(array)
    elif device=='0' or device=='1' or device=='2':
        with tf.device('/gpu:'+device): #use gpu
            convae = autoencoder_compiler_encoderonly(architecture,k1,k2,k3,number_of_features,dropout=dropout,) #build autoencoder model
            convae.load_weights(checkpoint,by_name=True) # load weights if wanted
            decoded_array=convae.predict(array)
    else:
        print('device error')
    return decoded_array

def features_of_single_array_with_model(array3D,model_path,device='CPU'):
    #takes an NORMALIZED (64,64,64) array and returns its feature vektor as array
    #compiles a new model! Don't use in a for-statement for multiple arrays
    #use predict_single_array_with_active_model

    array=prepare_single_array_for_predict(array3D)
    if device=='CPU':
        with tf.device('/cpu:0'): #only use CPU
            convae=tf.keras.models.load_model(model_path) #build autoencoder model
            decoded_array=convae.predict(array) 
    elif device=='0' or device=='1' or device=='2':
        with tf.device('/gpu:'+device):  #use gpu
            convae=tf.keras.models.load_model(model_path) #build autoencoder model
            decoded_array=convae.predict(array)
    else:
        print('device error')
    return decoded_array

def features_of_array_path_list_with_checkpoint(array_path_list,checkpoint,architecture=1,k1=16,k2=32,k3=64,number_of_features=16,dropout=0,device='CPU'):
    #takes a list of array paths and returns the feature vektors as an array of shape (list_len,feature_number)
    #compiles a new model! Don't use in a for-statement for multiple array_path_lists
    #use predict_array_path_list_with_active_model

    array=prepare_array_path_list_for_predict(array_path_list)
    if device=='CPU':
        with tf.device('/cpu:0'): #only use CPU
            convae = autoencoder_compiler_encoderonly(architecture,k1,k2,k3,number_of_features,dropout=dropout) #build autoencoder model
            convae.load_weights(checkpoint,by_name=True) # load weights if wanted
            decoded_array=convae.predict(array)
    elif device=='0' or device=='1' or device=='2':
        with tf.device('/gpu:'+device):  #use gpu
            convae = autoencoder_compiler_encoderonly(architecture,k1,k2,k3,number_of_features,dropout=dropout) #build autoencoder model
            convae.load_weights(checkpoint,by_name=True)# load weights if wanted
            decoded_array=convae.predict(array) 
    else:
        print('device error')
    return decoded_array

def features_of_array_path_list_with_model(array_path_list,model_path,device='CPU'):
    #takes a list of array paths and returns the feature vektors as an array of shape (list_len,feature_number)
    #compiles a new model! Don't use in a for-statement for multiple array_path_lists
    #use predict_single_array_with_active_model

    array=prepare_array_path_list_for_predict(array_path_list)
    if device=='CPU':
        with tf.device('/cpu:0'): #only use CPU
            convae=tf.keras.models.load_model(model_path) #build autoencoder model
            decoded_array=convae.predict(array) 
    elif device=='0' or device=='1' or device=='2':
        with tf.device('/gpu:'+device):  #use gpu
            convae=tf.keras.models.load_model(model_path) #build autoencoder model
            decoded_array=convae.predict(array) 
    else:
        print('device error')
    return decoded_array


def predict_array_path_list_with_active_model(array_path_list,active_model,device='CPU'):
    #takes a list of single .npy files, loads, normalizes and predicts them into an array

    array=prepare_array_path_list_for_predict(array_path_list)
    if device=='CPU':
        with tf.device('/cpu:0'): #only use CPU
            decoded_array=active_model.predict(array) 
    elif device=='0' or device=='1' or device=='2':
        with tf.device('/gpu:'+device):  #use gpu
            decoded_array=active_model.predict(array) 
    else:
        print('device error')
    return decoded_array

def predict_single_array_with_active_model(array3D,active_model,device='CPU'):
    #takes an NORMALIZED (64,64,64) array and returns its feature vektor as array

    array=prepare_single_array_for_predict(array3D)
    if device=='CPU':
        with tf.device('/cpu:0'): #only use CPU
            decoded_array=active_model.predict(array) # load weights if wanted
    elif device=='0' or device=='1' or device=='2':
        with tf.device('/gpu:'+device):  #use gpu
            decoded_array=active_model.predict(array) # load weights if wanted
    else:
        print('device error')
    return decoded_array

def features_from_combined_pandas_plus_greyboxes_to_pandas(pandas_single_file_path,greyboxes_data_path,result_file,active_model,device='CPU',Boxsize=64,AEversion='NA'):
    #Pandas with various DMC entries
    with open(pandas_single_file_path, "rb") as bb_data:
        boxes_total=pickle.load(bb_data)
        #boxes_total=boxes_total.loc[boxes_total['DMC'] == "26628261111219"]
    #get feature number from active_model
    greyboxes_path_list=gbp.get_greyboxes_paths(greyboxes_data_path,OC_version=boxes_total["OCversion"].iloc[0],size=Boxsize)[0:1]
    number_features=predict_array_path_list_with_active_model(greyboxes_path_list[0:1],active_model,device='CPU').shape[1]
    
    #create empty np.array to append all features to
    total_features=np.empty((0,number_features))
    print(total_features.shape)
    
    #get DMC List and OCversion from pandas dataset
    DMC_list=boxes_total["DMC"].unique()
    OCversion=boxes_total["OCversion"].unique()[0]
    #go through all DMCs in pandas list
    k=0
    for i in DMC_list:
        print(k+1,"-ter DMC with code: ", i)
        k=k+1
        ID_list=list(boxes_total.loc[boxes_total['DMC'] == i,"Box-ID"])
        print(boxes_total.loc[boxes_total['DMC'] == i,"Box-ID"])
        #ID_list=list(boxes_total.loc[boxes_total['DMC'] == "26134570111219"]["Box-ID"])
        print("Number of greyboxes for this DMC found in dataset: ", len(ID_list))
        #boxes_to_process_list=gbp.get_greyboxes_paths(greyboxes_data_path,DMC=i,OC_version=boxes_total["OCversion"].iloc[0],size=Boxsize,ID=ID_list)
        boxes_to_process_list=gbp.get_greyboxes_paths(greyboxes_data_path,DMC=i,size=Boxsize,ID=ID_list)
        slicing_point_count=math.floor(len(boxes_to_process_list)/1000)
        print("slicing point count: ", slicing_point_count)
        print("total len boxes to process:", len(boxes_to_process_list))
        for j in range(0,slicing_point_count):
            features=predict_array_path_list_with_active_model(boxes_to_process_list[j*1000:(j+1)*1000],active_model,device=device)
            print(features.shape)
            print(j)
            total_features=np.append(total_features,features,axis=0)
            print(total_features.shape)
        print("len boxes to process after for loop:", len(boxes_to_process_list[slicing_point_count*1000:]))
        features=predict_array_path_list_with_active_model(boxes_to_process_list[slicing_point_count*1000:],active_model,device=device)
        print(features.shape,total_features.shape)
        total_features=np.append(total_features,features,axis=0)
        del(ID_list,boxes_to_process_list,features)
        ####
    column_list=[]
    for i in range(1,number_features+1):
        column_list.append("feature"+f"{i:02d}")
    df=pd.DataFrame(total_features,columns=column_list)
    df.insert(0,'AEversion',AEversion)
    print(df)
    print(boxes_total)
    boxes_total=boxes_total.reset_index(drop=True)
    print(boxes_total)
    result=pd.concat([boxes_total,df],axis=1)
    print(result)
    result.to_pickle(result_file)

'''RANDOM SAMPLE GENERATORS'''

def create_random_samples_from_file(data,len_data=0,nr_samples=1,dtype=np.single):
    #creates a list of samples directly from a file with many arrays
    #data is path to a single file with many arrays

    if len_data==0:
        len_data=len(data) #costs a lot of time, if possible, give len(data) to function as constant
    sample=random.sample(range(0,len_data),nr_samples) #create list with random sample positions in the range of the train_sample_data
    return list(np.load(data)[sample].astype(dtype=dtype, copy=False)) #return created train sample as list

def create_random_samples_from_data_array(array,len_array=0,nr_samples=1,dtype=np.single):
    #creates a list of samples from a already loaded data array
    #array is an array of shape [samples,dim1,dim2,dim3]

    if len_array==0:
        len_array=array.shape[0] #costs a lot of time, if possible, give len(data) to function as constant
    sample=random.sample(range(0,len_array),nr_samples) #create list with random sample positions in the range of the train_sample_array
    return list(data[sample,:,:,:]) #return created train sample as list

def create_random_samples_from_data_list(datalist,len_list=0,nr_samples=1,dtype=np.single):
    #creates a list of samples from a list of filepaths
    #IN USE FOR TRAINING

    if len_list==0:
        len_list=len(datalist) #costs a lot of time, if possible, give len(data) to function as constant
    sample=random.sample(range(0,len_list),nr_samples) #create list with random sample positions in the range of the train_sample_list
    return itemgetter(*sample)(datalist)
    #return map(datalist.__getitem__, sample) #return created train sample

def splitlist(mainlist,nr_entries_new_list=1):
    #splits the given mainlist into two lists
    #the entries of the new sublist are taken out of the mainlist
    #the mainlist is reduced by all the elements, which are then part of the new sublist
    #returns two lists! first is mainlist, second is newlist
    #use as mainlist,sublist=splitlist(old_mainlist,nr_entries_transfered_into_new_list)

    sample=random.sample(range(0,len(mainlist)),nr_entries_new_list) #create random sample list of entries in the mainlist
    sublist=[] #initiate sublist
    for i in sorted(sample, reverse=True): #transfer all random samples from mainlist to sublist
        sublist.append(mainlist.pop(i))
    return mainlist,sublist #return two lists
    
def npy_header_offset(npy_path):
    #reads the header length of a numpy file

    #getting complicated here
    #.npy files need to be read in binary into a tensorflow.data.Dataset
    #because np.load is a numpy function with which a gradient can't be calculated during training
    #for this, the header length is needed and calculated by this function
    
    #takes a numpy filepath as input
    #returns the length of the header
    #copied from the internet: https://stackoverflow.com/questions/48889482/feeding-npy-numpy-files-into-tensorflow-data-pipeline
    #author: ely
    #code created Feb 20 '18 at 16:14

    with open(str(npy_path), 'rb') as f:
        if f.read(6) != b'\x93NUMPY':
            raise ValueError('Invalid NPY file.')
        version_major, version_minor = f.read(2)
        if version_major == 1:
            header_len_size = 2
        elif version_major == 2:
            header_len_size = 4
        else:
            raise ValueError('Unknown NPY file version {}.{}.'.format(version_major, version_minor))
        header_len = sum(b << (8 * i) for i, b in enumerate(f.read(header_len_size)))
        header = f.read(header_len)
        if not header.endswith(b'\n'):
            raise ValueError('Invalid NPY file.')
        return f.tell()

def parse_fn(filenames,header_offset):
    #function to read numpy file in binary
    dtype = tf.uint16
    return tf.data.FixedLengthRecordDataset(filenames, 64*64*64 * dtype.size, header_bytes=header_offset)

'''Training function'''

def autoencoder_training_data_directory(working_directory,
                                    data_directory,
                                    ## batch_size*(nr_batches_per_epoch_train+nr_batches_per_epoch_val)=number of samples loaded into GPU memory
                                    ## don't go over 10.000 samples when using tf.data.Dataset, because data is prefetched and buffered (speed increase)
                                    ## don#t go over 10.000 samples when using array data (much slower than tf.data.Dataset)
                                    batch_size=10, #number of boxes bevor weights are recalulated 
                                    nr_batches_per_epoch_train=10, #number of batches per sub epoch
                                    nr_batches_per_epoch_val=2, #number of batches for validation per sub epoch
                                    #number of main epochs. 
                                    #Each main epoch a new random sample out of all greyboxes in greyboxes_data_path of size "number of smaples" is chosen
                                    main_epoch_nr=10,
                                    sub_epoch_nr=10, #epochs with the same samples bevor a new sample is chosen
                                    architecture=0, #chose between main architectures as found in AE_compile
                                    k1=16,k2=32,k3=64, #number of conv kernels in the three conv layers. decoding is mirrored (k1,k2,k3 then k3,k2,k1)
                                    number_of_features=16, #number of features in the feature layer
                                    dropout=0, #dropout bevor dense layers
                                    AE_version=0, #AE_version as string, any string alowed, is used to create file and directory names
                                    run_nr=0, #set run nummer as string, any string alowed, is used to create file and directory names
                                    restart_file=0, #if set, a checkpoint is loaded. Give checkpoint file
                                    output=False, #if True, more information is printed in console. if False, nearly no information is printed
                                    plot_loss=False, # if True, the complete loss graph is plotted at the end of training
                                    plot_samples=False, # if True, three samples in the form of three slices are plottet at the end of trainingr of kernels in first layer
                                    Boxsized=True,Boxsize=64,Boxsingle=True,BoxDMC=False,BoxOC_version=False,BoxID=False): #which greyboxes should be used for training

    ###### START OF CODE ######

    ### DIRECTORY CHECK
    # check if directory exists, create new directory if it doesn't exist ####
    if not os.path.exists(working_directory+ r'\version_'+AE_version+r'\run_'+str(run_nr)+r'\checkpoints'):
        os.makedirs(working_directory+ r'\version_'+AE_version+r'\run_'+str(run_nr)+r'\checkpoints')

    #### COUNT TIME ###
    start_time = datetime.now() #test
    #if output==True:
    #    print("Time for execution:", datetime.now()-start_time, "\nAnzahl") #testausgabe
    
    ### NEW SESSION ###
    K.clear_session()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95 # only use this fraction of GPU memory. Increases stability if below 1.
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)

    ### COMPILE AUTOENCODER MODEL ###

    if output==True:
        verbose=1 #information during training is printed
    else:
        verbose=0 #no information during training is printed
    #mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:1","/gpu:2"],cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()) # use all available gpus
    mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:1","/gpu:2"],cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()) # use all available gpus
    mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"],cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()) # use all available gpus
    #mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:1"],cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()) # use all available gpus
    #mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:2"],cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()) # use all available gpus
    #mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:1","/gpu:2"]) # use all available gpus
    with mirrored_strategy.scope():
        convae = autoencoder_compiler(k1,k2,k3,number_of_features,dropout=dropout,ver=architecture,verbosity=verbose) #build autoencoder model
    if not restart_file == 0:
            convae.load_weights(restart_file) # load weights if defined
    if output==True:
        convae.summary() #show model

    #### CALLBACKS ####

    checkpoint_path = (working_directory+ r'\version_'+AE_version+r'\run_'+str(run_nr)+r'\checkpoints\checkpoint_v'+AE_version+r'_r'+str(run_nr)+r'_{val_loss:.5f}.hdf5') #construct checkpoint path and name
    checkpoint_dir = os.path.dirname(checkpoint_path) #construct checkpoint path for callbacks

    #checkpoint creation
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                    save_weights_only=True,
                                                    save_freq='epoch', #save checkpoints each epoch
                                                    verbose=verbose)

    #callbacks
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=100, monitor='loss'), #performance
                tf.keras.callbacks.TensorBoard(log_dir='.\logs'), #log_dir='./logs'
                cp_callback]

    ##### Get file paths lists for train and test #####

    array_loss=np.array([]) #to save loss of subepochs to mainepochs
    array_val_loss=np.array([]) #to save val_loss of subepochs to mainepochs
    
    train_list=gbp.get_greyboxes_paths(data_directory,sized=Boxsized,size=Boxsize,single=Boxsingle,DMC=BoxDMC,OC_version=BoxOC_version,ID=BoxID) #create list of data (filepaths)

    random.shuffle(train_list) #shuffle list to create random list order

    nr_samples_train=nr_batches_per_epoch_train*batch_size #calculate number of train samples per epoch
    nr_samples_val=nr_batches_per_epoch_val*batch_size #calculate number of test samples per epoch
    nr_division=nr_samples_val/nr_samples_train #calculate share of test samples to train samples per epoch

    train_list,test_list=splitlist(train_list,round(len(train_list)*nr_division)) #divide total data list into separate train and test lists

    len_train=len(train_list) #calculate real length of train data list
    len_test=len(test_list) #calculate real length of test data list
    print("Number of training greyboxes:", len_train)
    print("Number of test greyboxes:", len_test)

    ##### TRAINING START #####

    start_time = datetime.now() #log time
    if output==True:
        print("Start of training:",start_time,"\n") 

    ### MAIN EPOCHS
    for i in range(0,main_epoch_nr):
        if output==True:
            print("Main epoch ",str(i+1),"of ", main_epoch_nr)
        
        ### create new sample in form of two tf.data.Datasets for train and test

        dtype = tf.uint16 #dtype of the original data in the file
        header_offset = npy_header_offset(train_list[0]) #calculate header length (all files have to have same header length!)
        
        ### train dataset
        start_time_creation=datetime.now() #special time stamp for creation benchmark
        sample_train=list(create_random_samples_from_data_list(train_list, nr_samples=nr_samples_train)) #random sample of file paths out of the train sample list
        print(len(sample_train))
        if output==True:
            print("random sample generated in: ", datetime.now()-start_time_creation)
        start_time_creation=datetime.now()
        dataset_train = tf.data.Dataset.from_tensor_slices(sample_train) #create a tf.data.Dataset with the random sample of file paths
        #main read in of file. 
        #parse_fn() reads the file as bytes 
        #interleave reads multiple files at the same time for speedup
        #tf.io.decode_raw transforms read binary strings into float (dtype)
        #tf.reshape reshapes Data into 64,64,64 arrays
        #tf.math.divide normalized unint16 data into range between 0 and 1
        #.map adds all those functions to each entry in the Dataset, parallel calls for speedup
        #.zip builds a Dataset of (x,y) or (sample,label), which is here the same.
        #.shuffle shuffles the (x,y) tuples in the Dataset to avoid ascending order of files in list from random.sample
        #.batch combines the given number of tuples to a single batch
        #.prefetch preloads the given number of batches into memory (2 means twice the needed memory!)
        #took me one whole day to figure this out
        dataset_train = dataset_train.interleave(lambda x: parse_fn(x,header_offset),cycle_length=10,num_parallel_calls=tf.data.experimental.AUTOTUNE).map(lambda s: tf.math.divide(tf.reshape(tf.io.decode_raw(s, dtype), (64,64,64)),65535),num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #dataset_train = dataset_train.interleave(lambda x: parse_fn(x,header_offset),cycle_length=10).map(lambda s: tf.math.divide(tf.reshape(tf.io.decode_raw(s, dtype), (64,64,64)),65535))
        dataset_train = tf.data.Dataset.zip((dataset_train,dataset_train)).shuffle(nr_samples_train).batch(batch_size).prefetch(2)
        if output==True:
            print("train sample generated in:", datetime.now()-start_time_creation)
        
        ### test dataset, for info see above. Same but with test file paths list
        start_time_creation=datetime.now()
        sample_test=list(create_random_samples_from_data_list(test_list, nr_samples=nr_samples_val))
        if output==True:
            print("random sample generated in: ", datetime.now()-start_time_creation)
        start_time_creation=datetime.now()

        dataset_test= tf.data.Dataset.from_tensor_slices(sample_test)
        #dataset_test = dataset_test.interleave(lambda x: parse_fn(x,header_offset),cycle_length=5).map(lambda s: tf.math.divide(tf.reshape(tf.io.decode_raw(s, dtype), (64,64,64)),65535))
        dataset_test = dataset_test.interleave(lambda x: parse_fn(x,header_offset),cycle_length=5,num_parallel_calls=tf.data.experimental.AUTOTUNE).map(lambda s: tf.math.divide(tf.reshape(tf.io.decode_raw(s, dtype), (64,64,64)),65535),num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset_test = tf.data.Dataset.zip((dataset_test,dataset_test)).shuffle(nr_samples_val).batch(batch_size).prefetch(2)
        if output==True:
            print("test sample generated in:", datetime.now()-start_time_creation)  

        ### train model ###
        history = convae.fit(dataset_train, #train td.data.dataset of (x,y)
                    epochs=sub_epoch_nr, #number of epochs bevore a new sample is created
                    #batch_size=batch_size, #is given by tf.data.dataset
                    shuffle=True,
                    validation_data=dataset_test, #validation tf.data.dataset of (x,y)
                    verbose=verbose, #show training in output terminal
                    callbacks=callbacks) #activate callbacks
        # list all data in history
        array_loss=np.concatenate((array_loss,history.history['loss'])) #save loss of subepochs to mainepochs, since history isn't saved from mainepoch to mainepoch. Could be implemented if needed
        array_val_loss=np.concatenate((array_val_loss,history.history['val_loss'])) #save val_loss of subepochs to mainepochs
        if output==True:
            print(array_loss[-0]) #show last loss
            print(array_val_loss[-0]) #show last val_loss
        #loss arrays are saved each mainepoch to prevent dataloss when training is stoped 
        np.save(working_directory+ r'\version_'+AE_version+r'\run_'+str(run_nr)+r'\loss_v'+AE_version+r'_r'+str(run_nr)+r'.npy',array_loss) #save loss as .npy array
        np.save(working_directory+ r'\version_'+AE_version+r'\run_'+str(run_nr)+r'\val_loss_v'+AE_version+r'_r'+str(run_nr)+r'.npy',array_val_loss) #save val-los as .npy array
    if output==True:
        print("Duration of complete training:", datetime.now()-start_time)  #time for complete training

    ##### SAVE MODEL AND LOSSES #####
    #models are easy to recreate with checkpoint and not saved every epoch
    convae.save(working_directory+ r'\version_'+AE_version+r'\run_'+str(run_nr)+r'\model_v'+AE_version+r'_r'+str(run_nr)+r'.hdf5',save_format='h5') #save as h5 file
    convae.save(working_directory+ r'\version_'+AE_version+r'\run_'+str(run_nr)+r'\model_v'+AE_version+r'_r'+str(run_nr)+r'.tf',save_format='tf') #save as tf file

    if plot_loss==True:
        ##### PLOT LOSSES #####
        plot_loss_func(array_loss,array_val_loss)

    ##### PLOT THREE RANDOM SAMPLES with three ENCODED/DECODED slices each #####
    if plot_samples==True:
        Array=np.zeros((3,64,64,64))
        k=0
        for i in list(create_random_samples_from_data_list(train_list,len_train,3)):
            Array[k,:,:,:]=np.load(i)
            k=k+1
        decoded_data=convae.predict(Array)
        for i in range(0,3):
            plot_sample_64(decoded_data[i,:,:,:],Array[i,:,:,:])