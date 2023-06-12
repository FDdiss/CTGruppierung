import autoencoder_functions as asf
import numpy as np

directory_path = 'D:/AnomalieKI' #working directory
Autoencoder_path= directory_path+'/Autoencoder' #where the checkpoints will be created
greyboxes_data_path = directory_path+'/data/04_greyboxes_npy' #where the greyboxes are located

loss_npy=np.load(r'D:\AnomalieKI\Autoencoder\version_7\run_3\loss_v7_r3.npy')
val_loss_npy=np.load(r'D:\AnomalieKI\Autoencoder\version_7\run_3\val_loss_v7_r3.npy')

asf.plot_loss_func(loss_npy,val_loss_npy)
       
'''packages needed

main
    -tensorflow_gpu
    -pandas
    -numpy
various
    -random
    -pickle
    -operator
    -datetime

Make sure you start vscode over anaconda navigator! Otherwise paths may not be correctly set!
jupiter notebook failed testing in my build (Fabian)
'''

