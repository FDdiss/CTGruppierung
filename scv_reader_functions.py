##EINGEBEN: path = Dateipfad
import os
import numpy as np
import cupy as cp
from numpy import save
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

#version=3

#Unterfunktionen

def metadata_reader(filepath,file,header_bytes=1024,DMC='NA',savepath=0):
    #Header auslesen
    metadata = pd.DataFrame({'CT_DMC': '', 'Datasize': '', 'xdim': '', 'ydim': '', 'zdim': ''}, index=[0]) #...in extra Dataframe
    fsize = os.stat(filepath) #Ermitteln der Dateigröße   
    file.seek(int((0)))
    header = file.read(header_bytes) #Header einlesen
    #Automatisches Auslesen der Dimensionen aus Header
    # z.B. x = b'\xc5\x05'
    xdim = int.from_bytes(header[12:14], byteorder='little')
    ydim = int.from_bytes(header[16:18], byteorder='little')
    zdim = int.from_bytes(header[20:22], byteorder='little')
    if not (savepath==0):
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        metadata.loc[0]=[DMC, fsize.st_size.__str__(), xdim, ydim, zdim] #...als Dataframe
        metadata.to_pickle(savepath+r'/metadata_'+str(DMC))# + str(i+1)) #...als Pickle
    return(xdim,ydim,zdim)

def get_scvfile_paths(path):
    #finde alle scv Dateien in dem Ordner mit Unterordnern
    scvfiles = [os.path.join(root, name)
                for root, dirs, files in os.walk(path)
                for name in files
                if name.endswith((".uint16_scv"))]
    return(scvfiles)

def remove_scv_duplicates(pathlist,output=False):
    #Duplikate entfernen, falls DMC mehrfach vorhanden
    copy_pathlist=pathlist*1
    savelist=[]
    k=0
    for i in copy_pathlist:
        savestring=i
        pathlist.remove(i)
        trig=0
        for j in range(0,len(pathlist)):
            #finde den zweiten \\ und den achten _ von hinten um den DMC auszulesen (kann man bestimmt optimieren)
            a=savestring.rfind('\\')
            a=savestring.rfind('\\',0,a)
            a=savestring.rfind('_',0,a)
            a=savestring.rfind('_',0,a)
            a=savestring.rfind('_',0,a)
            a=savestring.rfind('_',0,a)
            a=savestring.rfind('_',0,a)
            a=savestring.rfind('_',0,a)
            a=savestring.rfind('_',0,a)
            a=savestring.rfind('_',0,a)
            if savestring[a-23:a-9] == pathlist[j][a-23:a-9]:
                trig=1
                k=k+1
                break
        if trig==0:
            savelist.append(savestring)
    if output==True:
        print("Number of duplicates removed: ", k)
    return(savelist)

def get_array_file_paths(path,endstring=".npy"):
    #übergibt eine Liste der Pfade aller mit "endstring" endenden Dateien in dem Verzeichnis mit Unterverzeichnis
    #standardmäßig gibt es alle .npy filepaths zurükck
    array_files = [os.path.join(root, name)
                for root, dirs, files in os.walk(path)
                for name in files
                if name.endswith((endstring))]
    return(array_files)

def delete_background_from_array(array):
    #plt.imshow(array[300,:,:], cmap='gray')
    #plt.show()
    #plt.imshow(array[:,300,:], cmap='gray')
    #plt.show()
    #plt.imshow(array[:,:,300], cmap='gray')
    #plt.show()
    #print(array.shape)
    array=np.delete(array,np.all((array == 26214), axis=(1,2)),axis=0)
    #print(array.shape)
    #plt.imshow(array[300,:,:], cmap='gray')
    #plt.show()
    #plt.imshow(array[:,300,:], cmap='gray')
    #plt.show()
    #plt.imshow(array[:,:,300], cmap='gray')
    #plt.show()
    array=np.delete(array,np.all((array == 26214), axis=(0,2)),axis=1)
    #print(array.shape)
    #plt.imshow(array[300,:,:], cmap='gray')
    #plt.show()
    #plt.imshow(array[:,300,:], cmap='gray')
    #plt.show()
    #plt.imshow(array[:,:,300], cmap='gray')
    #plt.show()
    array=np.delete(array,np.all((array == 26214), axis=(0,1)),axis=2)
    #print(array.shape)
    #plt.imshow(array[300,:,:], cmap='gray')
    #plt.show()
    #plt.imshow(array[:,300,:], cmap='gray')
    #plt.show()
    #plt.imshow(array[:,:,300], cmap='gray')
    #plt.show()
    return array

def delete_background_for_all_arrays_in_path(path,savepath):
    array_list=get_array_file_paths(path)
    print(array_list)
    k=1
    for i in array_list:
        print("Array", k, "of", len(array_list))
        k=k+1
        if "CTbr" in i:
            continue
        array=np.load(i)
        array=delete_background_from_array(array)
        arrayname=os.path.basename(i)[3:]
        print("generated: ", savepath+r'/CTbr_'+arrayname)
        save(savepath+r'/CTbr_'+arrayname,array)
    
#CTbr_26134390111219

#### Hauptprogramm ####

def scv_reader(raw_data_path,raw_array_data_path=0,meta_data_path=0,overwrite=False,output=False,delete_background=False):

    header_bytes=1024
    if raw_array_data_path==0:
        raw_array_data_path=raw_array_data_path
    scv_files=get_scvfile_paths(raw_data_path)
    if output==True:
        print(f'Number of scv-files found: {len(scv_files)}')
    scv_files=remove_scv_duplicates(scv_files,output=output)
    raw_data_path_length=len(raw_data_path)
    if not os.path.exists(raw_array_data_path):
        os.makedirs(raw_array_data_path)
    
    #Prüfen ob zugehöriges .npy array existiert
    if overwrite==False:
        copy_scv_files=scv_files*1
        number_files=len(scv_files)
        k=0
        for i in range(0,number_files):
            #finde den zweiten \\ und den achten _ von hinten um den DMC auszulesen (kann man bestimmt optimieren)
            a=copy_scv_files[i].rfind('\\')
            a=copy_scv_files[i].rfind('\\',0,a)
            a=copy_scv_files[i].rfind('_',0,a)
            a=copy_scv_files[i].rfind('_',0,a)
            a=copy_scv_files[i].rfind('_',0,a)
            a=copy_scv_files[i].rfind('_',0,a)
            a=copy_scv_files[i].rfind('_',0,a)
            a=copy_scv_files[i].rfind('_',0,a)
            a=copy_scv_files[i].rfind('_',0,a)
            a=copy_scv_files[i].rfind('_',0,a)
            DMC=copy_scv_files[i][a-23:a-9] #read DMC
            for dir,sub_dirs, files in os.walk(raw_array_data_path): 
                for name in files: 
                    if name.endswith(DMC+'.npy'):
                        scv_files.remove(copy_scv_files[i])
                        k=k+1
    if output==True:
        if overwrite==False:
            print('Number of those already transformed in arrays: ', k)
        print('Number of CT-arrays going to be created: ', len(scv_files)) 
    
    #Umwandeln der scv files
    for i in range(0,len(scv_files)): 
        start_time = datetime.now()
        with open(scv_files[i], 'rb') as scv_file: 
            #finde den zweiten \\ und den achten _ von hinten um den DMC auszulesen (kann man bestimmt optimieren)
            a=scv_files[i].rfind('\\')
            a=scv_files[i].rfind('\\',0,a)
            a=scv_files[i].rfind('_',0,a)
            a=scv_files[i].rfind('_',0,a)
            a=scv_files[i].rfind('_',0,a)
            a=scv_files[i].rfind('_',0,a)
            a=scv_files[i].rfind('_',0,a)
            a=scv_files[i].rfind('_',0,a)
            a=scv_files[i].rfind('_',0,a)
            a=scv_files[i].rfind('_',0,a)
            DMC=scv_files[i][a-23:a-9] #read DMC
            xdim,ydim,zdim = metadata_reader(scv_files[i],scv_file,header_bytes,DMC=DMC,savepath=meta_data_path) #dimensionen aus metadatareader lesen, metadaten abspeichern, falls meta_dat_path=/=0
            Array3D = np.zeros(xdim*ydim*zdim).reshape(ydim,xdim,zdim) #Array erzeugen mit den Abmessungen
            scv_file.seek(int(header_bytes))#Header überspringen
            Array3D = np.fromfile(scv_file,dtype=np.uint16, count=xdim*ydim*zdim).reshape(xdim,ydim,zdim, order='F') #scv file einlesen
            Array3D = np.rot90(np.flip(Array3D,(0)),3) #scv file drehen, um "alte" Darstellung beizubehalten, da vorherige Zeile nun die Dimensionen in anderer Reihenfolge einliest
            if delete_background==True:
                Array3D = delete_background(Array3D)
        if output==True:
            print("Array ", str(i+1)," out of ", str(len(scv_files)), " assembled. Duration of array assembly:", datetime.now()-start_time)
        start_time = datetime.now()
        save(raw_array_data_path+r'/CT_'+DMC+r'.npy', Array3D) #Array abspeichern
        if output==True:
            print("Array ", str(i+1)," out of ", str(len(scv_files)), " saved. Duration of array save:", datetime.now()-start_time)
    if output==True:
        print("scv reader finished")

def compare_two_scv_files(filepath1,filepath2,header_bytes=1024,output=False,DMC='NA'):
    #Geschrieben um zu prüfen ob zwei scv files identisch sind
    #notwendig um zu zeigen, dass zwei scv_files mit selben DMC Code identisch sind (wenn in 2Join und GeneralEvaluation Unterordnern zwei zu finden sind)
    with open(filepath1, 'rb') as scv_file: 
        xdim,ydim,zdim = metadata_reader(filepath1,scv_file,header_bytes,DMC=DMC,savepath=0)
        Array3D1 = np.zeros(xdim*ydim*zdim).reshape(ydim,xdim,zdim)
        scv_file.seek(int(header_bytes))
        Array3D1 = np.fromfile(scv_file,dtype=np.uint16, count=xdim*ydim*zdim).reshape(xdim,ydim,zdim, order='F')
        Array3D1= np.rot90(np.flip(Array3D1,(0)),3)
    with open(filepath2,'rb') as scv_file: 
        xdim,ydim,zdim = metadata_reader(filepath2,scv_file,header_bytes,DMC=DMC,savepath=0)
        Array3D2 = np.zeros(xdim*ydim*zdim).reshape(ydim,xdim,zdim)
        scv_file.seek(int(header_bytes))
        Array3D2 = np.fromfile(scv_file,dtype=np.uint16, count=xdim*ydim*zdim).reshape(xdim,ydim,zdim, order='F')
        Array3D2 = np.rot90(np.flip(Array3D2,(0)),3)
        compare=np.array_equal(Array3D2,Array3D1)
    if output==True:
        print("Arrays are equal:", str(compare))
    return compare