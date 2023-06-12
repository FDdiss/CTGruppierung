import autoencoder_functions as asf 

#path to greyboxes, where three random samples are chosen
OC_greyboxes_data_path = 'D:/AnomalieKI/data/04_greyboxes_npy'

#set conv AE architecture as used by checkpoint model
architecture=1

#kernel numbers
k1=64
k2=32
k3=16

#path to the checkpoint file (not model file)
checkpoint_path=r"D:\AnomalieKI\Autoencoder\version_7\run_6\checkpoints\checkpoint_v7_r6_0.00007.hdf5"

#code
#needs all parameters to create model
#k1,k2,k3 are the number of kernels in the layers
#architecture is the architecture number as defined in AE_compile
asf.plotsamples_with_checkpoint(OC_greyboxes_data_path,checkpoint_path,k1=64,k2=32,k3=16,number_of_features=16,architecture=1,
                                                    Boxsized=True,Boxsize=64,Boxsingle=True,BoxDMC=False,BoxOC_version=False,BoxID=False)
#print(asf.get_greyboxes_paths(OC_greyboxes_data_path))

''' packages needed
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

