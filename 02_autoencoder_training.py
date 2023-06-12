import autoencoder_functions as asf 

directory_path = r'D:/AnomalieKI' #working directory
Autoencoder_path= directory_path+r'/Autoencoder' #where the checkpoints will be created
greyboxes_data_path = directory_path+r'/data/04_greyboxes_npy' #where the greyboxes are located

''' Versionhistory'''

asf.autoencoder_training_data_directory(Autoencoder_path,
                                greyboxes_data_path,
                                ## batch_size*(nr_batches_per_epoch_train+nr_batches_per_epoch_val)=number of samples loaded into GPU memory
                                ## don't go over 10.000 samples
                                batch_size=1, #number of boxes bevor weights are recalulated 
                                nr_batches_per_epoch_train=8, #number of batches per sub epoch
                                nr_batches_per_epoch_val=2, # number of batches for validation per sub epoch
                                #number of main epochs. 
                                #Each main epoch a new random sample out of all greyboxes in greyboxes_data_path of size "number of smaples" is chosen
                                main_epoch_nr=100,
                                sub_epoch_nr=5, #epochs with the same samples bevor a new sample is chosen
                                architecture=1, #chose between main architectures as found in asf
                                k1=64,k2=32,k3=16, #number of conv kernels in the three conv layers. decoding is mirrored (k1,k2,k3 then k3,k2,k1)
                                dropout=0.2,#dropout bevor dense layers
                                AE_version="test",#set your AE_version as string, any string alowed
                                run_nr="1", #set run nummer as string, any string alowed
                                restart_file=r"D:\AnomalieKI\Autoencoder\version_7\run_5\checkpoints\checkpoint_vVersion7_r5_0.00011.hdf5",
                                output=True, #if True, more information is printed in console. if False, nearly no information is printed
                                plot_loss=True, # if True, the complete loss graph is plotted at the end of training
                                plot_samples=True) # if True, three samples in the form of three slices are plottet at the end of training

#asf.plotsamples_with_checkpoint(OC_greyboxes_data_path,r"D:\AnomalieKI\Autoencoder\version_6\run_1\checkpoints",16,32,64,0.2)


'''
Version7
ARCHITECTURE 1: 
    reduced feature layer to 16
    added dense layers bevore and after featurelayer, 
    added dropout to those new dense layer, 
    deleted spatial dropout,
    implemented padding=same for all conv and softpooling layers,
    changed kernel size to 8,8,8 of all conv layers

        r1: architecture 1,ten CT Scans data basis, 50main/10sub epochs, 250batch, 80/20 batches per epoch, k(64,32,16), dropout0.2, restart from scratch
        r2: changed back to mean sqaure error, stoped for server maintenance after 20h
        added .shuffle(nr_samples_train/val) to datasets
        r3: restart from v7r2, checkpoint_v7_r2_0.00012.hdf5
        run 4: restart from v7r3
        run 5: restart from v7r4 D:\AnomalieKI\Autoencoder\version_7\run_4\checkpoints\checkpoint_v7_r4_0.00008.hdf5
        run 6: restart from v7r5 D:\AnomalieKI\Autoencoder\version_7\run_5\checkpoints\checkpoint_vVersion7_r5_0.00011.hdf5

        '''