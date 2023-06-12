import cupy as cp
import numpy as np
import time
import os
import pickle
from numpy import load
from numpy import save
import pandas as pd
import matplotlib.pyplot as plt
import cupyx.scipy.ndimage as cnd
import scipy.ndimage
from datetime import datetime

#Testfunktion zum prüfen,ob richtig importiert
def test():
    print('Herzlichen Glückwunsch du hast richtig importiert')
    
#Kugel Funktionen für strukturierendes Element
def Kugel_r2_cupy(radius):#Kugelgleichung:x^2+y^2+z^2=r^2 ==>
    r=radius
    A = cp.arange(-r,r+1)**2 #x^2
    B=A[:,None]+A #x^2+y^2, Kreis
    C=B[:,:,None]+A #x^2+y^2+z^2; Kugel
    #return cp.where(C<=r**2+1,1,0)
    return cp.where(C<=r**2,1,0)

def Kugel_r_cupy(radius):#Kugelgleichung:x^2+y^2+z^2=r^2 ==>
    r=radius
    A = cp.arange(-r,r+1)**2 #x^2
    B=A[:,None]+A #x^2+y^2, Kreis
    C=B[:,:,None]+A #x^2+y^2+z^2; Kugel
    D=cp.sqrt(C)
    return cp.where(D-r<=0.1,1,0)#Wert für <= zu wälen

#Funktion zur Kontrastnormierung
def kontnorm(Array3D):
    #Kontrastnormierug 
    gmin=Array3D.min()#globales Grauwertminimum
    gmax=Array3D.max()#globales Grauwertmaximum
    a=65535/(gmax-gmin)
    #print(cp.amin(a*(Array3D-gmin)), "minimum Arrayvalue, should be greater than 0")
    return (a*(Array3D-gmin)).astype('uint16')
    #return (a*Array3D+b).astype('uint16')#Rückgabe des normierten Grauertarrays als uint16
    #return (a*Array3D+b)#Rückgabe des normierten Grauertarrays als uint16

#Funktion für Bundingboxes um Fehler
def bounding_boxes_OC(Array3D,structure,norm=True,footprint=True,seg_falue=9200,struc_labels=cp.ones([3,3,3]),output=False):
    
    ##########Beschreibung Übergabeparameter##########
    '''
    Array3D: Grauwertarray auf dem Fehler gefunden werden sollen (numpy cupy oder pandas)
    
    structure: strukturierendes Element, mit dem Opening Closing durchgeführt werden soll (numpy cupy oder pandas)
    
    norm: Kontrastnormierung auf Array3D norm=True; keine Kontrastnormierung norm=False; default: True
    
    footprint: structure als footprint oder als structure anOpening Closing übergeben; footprint=true als footprint übergeben; default True
    
    seg_falue: Segmentierungsschwellwert, ab dem Differenz segmentiert wird; default 9200
    
    struc_labels: strukturierendes Element, das die Nachbarschaft beschreibt, für Labeling; default struc_labels=cp.ones([3,3,3])
    '''    
    ##################################################
    
    #Datentyp des 3DArrays abfragen und falls notwendig in cupy ändern
    typ_Array=str(type(Array3D))
    if 'pandas' in typ_Array:    
        Array3D=cp.asarray(Array3D.to_numpy())
    elif 'numpy' in typ_Array:
        Array3D=cp.asarray(Array3D)
        #Datentyp des strukturierenden Elements abfragen und falls notwendig in cupy ändern
    typ_struc=str(type(structure))
    if 'pandas' in typ_struc:    
        structure=cp.asarray(structure.to_numpy())
    elif 'numpy' in typ_struc:
        structure=cp.asarray(structure)
    #Kontrastnormierung wenn gewünscht
    if norm:
        Array3D=kontnorm(Array3D)
        
    #wenn footprint=true Opening Closing mit footprint
    if footprint:
        if output==True:
            print(" Now processing opening with footprint", datetime.now())
            start_time=datetime.now()
        Cuda_Array_o=cnd.grey_opening(Array3D,footprint=structure).astype('uint16') #GPU opening
        #print(cp.amin(Cuda_Array_o), "min Cuda opening")
        opening=cp.asnumpy(Cuda_Array_o) #Ergebnis in Speicher zurückspeichern, braucht Zeit aber sonst zu wenig Platz auf GPU
        del Cuda_Array_o #Speicher freigeben

        if output==True:
            print("     Duration for opening: ", datetime.now()-start_time)
            print(" Now processing closing with footprint", datetime.now())
            start_time=datetime.now()
        Cuda_Array_c=cnd.grey_closing(Array3D,footprint=structure) #GPU Closing
        #print(cp.amin(Cuda_Array_c), "min Cuda closing")
        closing=cp.asnumpy(Cuda_Array_c) #Ergebnis in Speicher zurückspeichern, braucht Zeit aber sonst zu wenig Platz auf GPU
        del Cuda_Array_c #Speicher freigeben 

    #sonst Opening Closing mit structure
    else:
        if output==True:
            print(" Now processing opening with structure", datetime.now())
            start_time=datetime.now()
        Cuda_Array_o=cnd.grey_opening(Array3D,structure=structure) #GPU opening
        if output==True:
            print("     Duration for opening: ", datetime.now()-start_time)
            print(" Now processing closing with structure", datetime.now())
            start_time=datetime.now()
        Cuda_Array_c=cnd.grey_closing(Array3D,structure=structure) #GPU Closing

    #Differenz von Opening und Closing bilden
    if output==True:
        print("     Duration for closing: ", datetime.now()-start_time)
        print(" Now calculating difference", datetime.now())
        start_time=datetime.now()
    diff=np.empty

    #diff=Cuda_Array_c-Cuda_Array_o
    diff=closing-opening

    #del(Cuda_Array_c,Cuda_Array_o,Array3D)
    del (closing, opening, Array3D) # Speicher freigeben
    
    #Segmentierung 
    if output==True:
        print("     Duration for calculating difference: ", datetime.now()-start_time)
        print(" Now calculating segmentation", datetime.now())
        start_time=datetime.now()
    diff=cp.asarray(diff)

    seg=cp.where(diff>seg_falue,1,0)
    del(diff) # speicher freigeben

    if output==True:
        print("     Duration for segmentation step: ", datetime.now()-start_time)
        print(" Now labeling data", datetime.now())
        start_time=datetime.now()
    
    #Labeling
    [labels,num]=cnd.label(seg,structure=struc_labels)

    del(seg) #Speicher freigeben

    #Bounding boxes erzeugen
    if output==True:
        print("     Duration for labeling: ", datetime.now()-start_time)
        print(" Now generating boxes", datetime.now())
        start_time=datetime.now()
    chunklist=[]
    Chunks=np.zeros([num,10], dtype=int) #leeres Df für Maximal und Minimalwerte festlegen
    for x in range(1,(num+1)): #für alle Cluster
        x_list=np.any(labels == x, axis=(0,2)).tolist()
        y_list=np.any(labels == x, axis=(1,2)).tolist()
        z_list=np.any(labels == x, axis=(0,1)).tolist()
        #Minimale und maximale Koordinaten speichern
        Chunks[x-1,1]=x_list.index(True)
        Chunks[x-1,2]=len(x_list)-x_list[::-1].index(True) - 1
        Chunks[x-1,3]=y_list.index(True)
        Chunks[x-1,4]=len(y_list)-y_list[::-1].index(True) - 1
        Chunks[x-1,5]=z_list.index(True)
        Chunks[x-1,6]=len(z_list)-z_list[::-1].index(True) - 1
        Chunks[x-1,0]=cp.asnumpy(structure.shape[0]) #Größe des strukturierenden Elements speichern
         #Boxgrößen in Dimensionen bestimmen
        Chunks[x-1,7]=cp.asnumpy(Chunks[x-1,2]-Chunks[x-1,1]+1)
        Chunks[x-1,8]=cp.asnumpy(Chunks[x-1,4]-Chunks[x-1,3]+1)
        Chunks[x-1,9]=cp.asnumpy(Chunks[x-1,6]-Chunks[x-1,5]+1)
    if output==True:
        print("     Boxes generated:", str(num))
        print("     Duration for generating boxes: ", datetime.now()-start_time)
    return pd.DataFrame(Chunks, columns=['strucsize', 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax','xsize','ysize','zsize']) #...in extra Dataframe
    del(Chunks,indexes,labels,diff) #Speicher freigeben


#Funktion zum iterativen Aufrufen von Opening Cosing in zu großen Clustern mit Abbruch erst bei kleinstem Element
def error_bounding_min(Array3D,struc_größe,structure='Kugel',würfelgröße=50,step=2,norm=True,footprint=True,seg_falue=9200,struc_labels=cp.ones([3,3,3]),error_min=[3,3,3]):
    '''
    Array3D: Grauwertarray auf dem Fehler gefunden werden sollen (numpy cupy oder pandas)
    
    structure:  'Kugel': erzeugt Kugel als strukturirendes Element
                'Würfel' erzeugt Würfel als strukturirendes Element
                ; Angabe von struc_größe nötig
    
    struc_größe: Angabe wie groß das erste erzeugte strukturierendes Element sein soll; nur ungerade Zahlen! Da bei Bei Kugel Nullpunkt mitberücksichtigt
    
    würfelgröße: bis zu welcher Größe sollen die Boundingboxes verkleinert werden
    
    step: in welcher Schritteite werden die strukturierenden Elemente verkleinert
    
    norm: Kontrastnormierung auf Array3D norm=True; keine Kontrastnormierung norm=False; default: True
    
    footprint: structure als footprint oder als structure anOpening Closing übergeben; footprint=true als footprint übergeben; default True
    
    seg_falue: Segmentierungsschwellwert, ab dem Differenz segmentiert wird; default 9200
    
    struc_labels: strukturierendes Element, das die Nachbarschaft beschreibt, für Labeling; default struc_labels=cp.ones([3,3,3])
    
    error_min: minimale Größe einesvermeintlichen Fehlers die berücksichtigt werden soll; [xdim,ydim,zdim]; default[3,3,3]
    '''
    #Überprüfen, ob ungerade Größe gegeben
    if struc_größe%2:
    
        struc_min=3 # kleinste sinnvolle Größe für Kugel

        #Datentyp des 3DArrays abfragen und falls notwendig in cupy ändern
        typ_Array=str(type(Array3D))
        if 'pandas' in typ_Array:    
            Array3D=cp.asarray(Array3D.to_numpy())
        elif 'numpy' in typ_Array:
            Array3D=cp.asarray(Array3D)

        #Structure abfragen und entweder Kugel oder Würfel erzeugen 
        größe_struc=struc_größe
        if structure=='Kugel':
            struc=Kugel_r_cupy((größe_struc-1)/2)#-1, um gerade größe für Radius zu erhalten
        elif structure=='Würfel':
            struc=cp.ones((größe_struc,größe_struc,größe_struc))#Würfel

        #erster Durchlauf mit voller Größe des strukturierenden Elements
        errorboxes=bounding_boxes_OC(Array3D,struc,norm,footprint,seg_falue,struc_labels)

        #leeres Datframe für neue Bouningboxes erzeugen
        alt=pd.DataFrame([], columns=['strucsize', 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax','xsize','ysize','zsize']) 

        #von der zweit größten Größe bis zur minimalen Größe im Abstand step... 
        for größe_struc in range(struc_größe-step,struc_min,-step):            
            #leeres Datframe für neue Bouningboxes erzeugen
            neu=pd.DataFrame([], columns=['strucsize', 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax','xsize','ysize','zsize']) 
            #neues verkleinertes strukturierendes Element für neuen Durchlauf OpeningClosing anlegen nur einmal nicht für alle Zeilen extra 
            if structure=='Kugel':
                struc=Kugel_r_cupy((größe_struc-1)/2)#-1, um gerade größe für Radius zu erhalten
            elif structure=='Würfel':
                struc=cp.ones((größe_struc,größe_struc,größe_struc))#Würfel
            
            #Alle Zeilen von errorboxes auswählen, die in allen Dimensionen größer als 3 mal das strukturierende Element sind
            ''' 3 als grobe abschätzung evtl anpassen'''
            to_big=errorboxes[(errorboxes['xsize']>(größe_struc)*3) | (errorboxes['ysize']>(größe_struc)*3)| (errorboxes['zsize']>(größe_struc)*3)]
            #für alle Indexe in den gewähten Zeilen
            for y in to_big.index:
                #Größe der Boundingbox um halbe strukturierendes Element in jede Richtung vergrößern
                #wenn dann kleiner 0, auf 0 setzen
                #wenn dann größer als Dimension, auf Maximum setzen (Dimension-1, um letzen Index zu wählen)
                #[0]= y-Achse
                #[1]= x-Achse
                #[2]= z-Achse                
                xmin=int(errorboxes['xmin'][y]-(größe_struc-1)/2)
                if xmin<0:
                    xmin=0
                xmax=int(errorboxes['xmax'][y]+(größe_struc-1)/2)
                if xmax>Array3D.shape[1]-1:
                    xmax=Array3D.shape[1]-1
                ymin=int(errorboxes['ymin'][y]-(größe_struc-1)/2)
                if ymin<0:
                    ymin=0
                ymax=int(errorboxes['ymax'][y]+(größe_struc-1)/2)
                if ymax>Array3D.shape[0]-1:
                    ymax=Array3D.shape[0]-1
                zmin=int(errorboxes['zmin'][y]-(größe_struc-1)/2)
                if zmin<0:
                    zmin=0
                zmax=int(errorboxes['zmax'][y]+(größe_struc-1)/2)
                if zmax>Array3D.shape[2]-1:
                    zmax=Array3D.shape[2]-1
                    
                #Grauwertarray erzeugen mit der angepasten Boundingbox der gewählten Zeile aus errorboxes und ursprünglichen Grauwerten
                Array3D_neu=Array3D[ymin:ymax+1,xmin:xmax+1,zmin:zmax+1]#+1 für Indexierng
                #neue boundingboxes auf Subarray erzeugen und in neu abspeichern
                neu=neu.append(bounding_boxes_OC(Array3D_neu,struc,norm,footprint,seg_falue,struc_labels))
                #Zeile mit zu großer Boundingbox in alt speichern
                alt=alt.append(errorboxes[errorboxes.index==y])
                #gewählte Zeile mit zu großer Bundingbox über Index aus errorboxes löschen
                errorboxes.drop([y], axis=0,inplace=True)

            #neue Boundingboxes an errorboxes anhängen
            errorboxes=errorboxes.append(neu)
            #indexe neu setzen
            errorboxes.reset_index(drop=True,inplace=True)
        #Duplikate löschen
        errorboxes.drop_duplicates(subset=['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax','xsize','ysize','zsize'],inplace=True)
        #zu kleine Boundingboxes abfragen
        zu_klein=errorboxes[(errorboxes['xsize']<error_min[0])| (errorboxes['ysize']<error_min[1]) | (errorboxes['zsize']<error_min[2])]
        #zu kleine Boundingboxes löschen
        errorboxes.drop(zu_klein.index,axis=0,inplace=True)
         #indexe neu setzen
        errorboxes.reset_index(drop=True,inplace=True)
        #Boundingboxes der Fehler, ersetzte Boundingboxes  und zu kleine Boundingboxes zurückgeben  
        return errorboxes, alt, zu_klein
    
    #wenn gerade größe für strukturierendes Element gegeben
    else:
        print('Fehler: bitte nur ungeade größen für strukturierendes Element, dabei Kugelmittelpunkt berücksichtigt')
        
#Funktion zum erzeugen Bpundingboxes bei gesmtes Array für alle Elemente zu durchlaufen
def error_bounding_total(Array3D,struc_größe,structure='Kugel',step=2,norm=True,footprint=True,seg_falue=9200,struc_labels=cp.ones([3,3,3]),error_min=[3,3,3],output=False,minstruc=3):
    
    '''
    Array3D: Grauwertarray auf dem Fehler gefunden werden sollen (numpy cupy oder pandas)
    
    structure:  'Kugel': erzeugt Kugel als strukturirendes Element
                'Würfel' erzeugt Würfel als strukturirendes Element
                ; Angabe von struc_größe nötig
    
    struc_größe: Angabe wie groß das erste erzeugte strukturierendes Element sein soll; nur ungerade Zahlen! Da bei Bei Kugel Nullpunkt mitberücksichtigt
    
    step: in welcher Schritteite werden die strukturierenden Elemente verkleinert
    
    norm: Kontrastnormierung auf Array3D norm=True; keine Kontrastnormierung norm=False; default: True
    
    footprint: structure als footprint oder als structure anOpening Closing übergeben; footprint=true als footprint übergeben; default True
    
    seg_falue: Segmentierungsschwellwert, ab dem Differenz segmentiert wird; default 9200
    
    struc_labels: strukturierendes Element, das die Nachbarschaft beschreibt, für Labeling; default struc_labels=cp.ones([3,3,3])
    
    error_min: minimale Größe einesvermeintlichen Fehlers die berücksichtigt werden soll; [xdim,ydim,zdim]; default[3,3,3]
    '''
    #Überprüfen, ob ungerade Größe gegeben
    if struc_größe%2:
    
        if minstruc<3:# kleinste sinnvolle Größe für Kugel
            minstruc=3
        #Datentyp des 3DArrays abfragen und falls notwendig in cupy ändern
        typ_Array=str(type(Array3D))
        if 'pandas' in typ_Array:    
            Array3D=cp.asarray(Array3D.to_numpy())
        elif 'numpy' in typ_Array:
            Array3D=cp.asarray(Array3D)
        #leeres Dataframe für Boundingboxes anlegen
        errorboxes=pd.DataFrame([], columns=['strucsize', 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax','xsize','ysize','zsize'])
        #von der gegeben Größe bis zur minimalen Größe im Abstand step... 
        for größe_struc in range(struc_größe,minstruc-1,-step):
            #Structure abfragen und entweder Kugel oder Würfel erzeugen
            start_time=datetime.now()
            if output==True:
                print("Now processing opening/closing with structure size: ",str(größe_struc))
            if structure=='Kugel':
                struc=Kugel_r_cupy((größe_struc-1)/2)#-1, um gerade größe für Radius zu erhalten
            elif structure=='Würfel':
                struc=cp.ones((größe_struc,größe_struc,größe_struc))#Würfel

            #opening Closing mit jeweiligem strukturierendem Element
            errorboxes=errorboxes.append(bounding_boxes_OC(Array3D,struc,norm,footprint,seg_falue,struc_labels,output=output))
            #print(errorboxes[(errorboxes['zmin']==200) & (errorboxes['zmax']==202)]) 
            if output==True:
                print("Finished with structure size",str(größe_struc),". Elapsed time: ", datetime.now()-start_time)

        #indexe neu setzen           
        errorboxes.reset_index(drop=True,inplace=True)

        #Duplikate löschen
        errorboxes.drop_duplicates(subset=['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax','xsize','ysize','zsize'],inplace=True,keep='first')

        #zu kleine Boundingboxes abfragen
        zu_klein=errorboxes[(errorboxes['xsize']<error_min[0])| (errorboxes['ysize']<error_min[1]) | (errorboxes['zsize']<error_min[2])]
        #zu kleine Boundingboxes löschen
        errorboxes.drop(zu_klein.index,axis=0,inplace=True)
        #indexe neu setzen
        errorboxes.reset_index(drop=True,inplace=True)
        #Boundingboxes der Fehler  und zu kleine Boundingboxes zurückgeben  
        return errorboxes
    
    #wenn gerade größe für strukturierendes Element gegeben
    else:
        print('Fehler: bitte nur ungeade größen für strukturierendes Element, dabei Kugelmittelpunkt berücksichtigt')


#Funktion zum auslesen der Grauwerte
def get_Grauwerte(Array3D,boundingboxes):
    '''
        Array3D: gesamtes Grauwertarray
        
        boundingboxes: auf dem Array3D erzeugte Boundingboxes
        
        retun: Liste mit Grauwertarrays
    '''
    Würfel=[] #leere Liste erzeugen, die mit Grauwertarrays gefüllt wird
    for x in boundingboxes.index:
        Würfel.append(Array3D[boundingboxes['ymin'][x]:boundingboxes['ymax'][x]+1,boundingboxes['xmin'][x]:boundingboxes['xmax'][x]+1,boundingboxes['zmin'][x]:boundingboxes['zmax'][x]+1])
    return Würfel  

####################################################


def boxreshape(array,desired_shape,interpolorder=1):
    #check if array dimension is same as length of desired_shape tuple
    if not(len(desired_shape)==array.ndim):
        raise Exception ("tuple length doesn't match array size")
    #calculating reshaping factors
    factortuple=()
    for i in range(len(desired_shape)):
        factortuple+=(desired_shape[i]/array.shape[i],)
    #reshape array
    desired_array=cnd.zoom(array,factortuple,order=interpolorder) 
    #returnvalue
    #return(desired_array.astype(type(array.flat[0])))
    return(desired_array.astype(array.dtype))

def create_OC_boundingboxes(raw_array_single_data_path,pandas_data_path=0,DMC='NA',gpu=0,maxstruc=43,output=False,minstruc=0,OCversion="99"):
    #Hauptfunktion
    #erzeugt die Bounding Boxes für das übergebene Array
    #speichert die Bounding Boxes in eine pandas file in den angebenen Ordner

    #maxstruc gibt den Beginn an für die Größe des strukturierenden Elements
    #minstruc gibt die letzte Größe an, für die das opening/closing durchgeführt wird. Wird auf 3 gesetzt, wenn ein Wert kleiner 3 übergeben wird


    #gpu gibt die GPU an, auf der gerechnet wird
    #für parallele verwendung die Funktion für jede GPU einzeln starten und die GPU Nummer durchwechseln

    #DMC übergibt den DMC, falls er nicht aus dem array path ausgelesen werden kann
    
    #output erlaubt benachmark und debugging durch mehr Informationen in der Konsole
        
    if pandas_data_path==0: #sollte kein Speicherort für die pandas Datei angegeben werden, wird sie in das gleiche Verzeichnis gespeichert wie das array
        pandas_data_path=os.path.dirname(os.path.realpath(raw_array_single_data_path))
    else: # falls ein Speicherort angegeben wurde, wird der Ordner erzeugt, falls nicht vorhanden
        if not (os.path.exists(pandas_data_path)):
            os.makedirs(pandas_data_path)
    
    if DMC=='NA' and len(raw_array_single_data_path)>=18: #falls kein DMC angegeben wurde, wird er aus dem arraynamen erzeugt, falls er zumindest so lange ist, dass man etwas auslesen kann, ansonsten 'NA'
        DMC=raw_array_single_data_path[-18:-4] #raw arrays enthalten normalerweise den DMC an diesen Stellen im Namen
    with cp.cuda.Device(gpu): #mit der angegebenen GPU rechnen
        Array3D_c = cp.load(raw_array_single_data_path) #Datei einlesen als cupy array
        
        #openingclosing
        start_time = datetime.now()
        if output==True:
            print("Starting at: ", start_time)
            print("generating bounding boxes")
        boxes_total=error_bounding_total(Array3D_c,struc_größe=maxstruc,output=output,minstruc=minstruc) #Hauptfunktion, ändert minstruc auf mindestens 3
        del Array3D_c #Speicher freigeben
        if output==True:
            print("bounding boxes generated in ", datetime.now()-start_time)
        boxes_total['DMC']=DMC #Abspeichern des DMC in das pandas file
        boxes_total['Box-ID']=boxes_total.index #Durchnummerieren der ID
        cp.get_default_pinned_memory_pool().free_all_blocks() #Speicher freigeben
        cp.get_default_memory_pool().free_all_blocks() #Speicher freigeben
    if minstruc<3: #ändert minstruc auf mindestens 3, ansonsten wird die angegebene Größe für den Dateinamen verwendet
        minstruc=3
    #Speichern
    #boxes_total.rename(columns ={"CT-Nummer":"DMC"},inplace=True)
    #boxes_total.rename(columns ={"Größe_structure":"strucsize"},inplace=True)
    #boxes_total.rename(columns ={"x_Größe":"xsize"},inplace=True)
    #boxes_total.rename(columns ={"y_Größe":"ysize"},inplace=True)
    #boxes_total.rename(columns ={"z_Größe":"zsize"},inplace=True)
    boxes_total.insert(10,'OCversion',OCversion)
    boxes_total["Box-ID"]=boxes_total["Box-ID"].apply(lambda x: f'{x:05d}')
    boxes_total.to_pickle(pandas_data_path+r'/pandas_bb_'+OCversion+r"_"+DMC+r'_max.s_'+f"{maxstruc:02d}"+r'_min.s_'+f"{minstruc:02d}")
    del(boxes_total) #Speicher freigeben
    cp.get_default_memory_pool().free_all_blocks()#Speicher freigeben
    cp.get_default_pinned_memory_pool().free_all_blocks()#Speicher freigeben

def create_greyboxes_with_bb(raw_array_single_data_path,pandas_data,OC_greyboxes_data_path=0,resize_size=0,save_separate=False,gpu=0):
    #Erzeugt aus array und pandas file Greyboxes
    #Skaliert optional die Boxen auf gegebene Größe
    #Speichtert skalierte und unskalierte Greyboxes ab
    #Speichern erfolgt entweder in eine gemeinsame Array-Datei(save_separate=False) für alle unskalierten bzw. für alle skalierten Greyboxes
    #oder alternativ jede Box in eine eigene Datei (save_separate=True)
    #rechnet auf der GPU, die angegeben ist
    #geht sehr schnell
    print(pandas_data)
    DMC=pandas_data['DMC'].iloc[0] #Auslesen des DMC aus dem pandas file
    OCversion=pandas_data['OCversion'].iloc[0] #Auslesen des DMC aus dem pandas file
    #Speicherpfad überprüfen und notfalls erzeugen
    if OC_greyboxes_data_path==0: #wenn kein Speicherpfad angegeben, ist der Array-Ordner der Speicherpfad
        OC_greyboxes_data_path=os.path.dirname(os.path.realpath(raw_array_single_data_path))
    else:
        if not (os.path.exists(OC_greyboxes_data_path+r'/'+str(DMC))):
            os.makedirs(OC_greyboxes_data_path+r'/'+str(DMC))
    
    with cp.cuda.Device(gpu):
        #openingclosing
        start_time = datetime.now()
        Array3D_c = cp.load(raw_array_single_data_path) #einladen des Arrays als cupy Array
        grey=get_Grauwerte(Array3D_c,pandas_data) #Erzeugen der Greyboxes
        del Array3D_c #Speicher freigeben
        #Skalieren
        if not (resize_size==0): #wenn Skalierungsgröße angegeben
            desired_shape=(resize_size,resize_size,resize_size)
            if save_separate==False: #Wenn Gesamtdatei gewünscht, alle Originalarrays in einer Datei abspeichern
                filename=OC_greyboxes_data_path+r'/greyboxes_'+OCversion+r"_unsized_"+DMC+r'.npy'
                if not os.path.isfile(filename):
                    save(filename,cp.asnumpy(grey))
            #Skalieren aller Greyboxes
            for x in range(0,len(grey)): 
                if save_separate==True: #Wenn Einzeldateien gewünscht, jeweiliges Originalarray abspeichern
                    filename=OC_greyboxes_data_path+r'/'+str(DMC)+r'/greybox_'+OCversion+r'_unsized_'+DMC+r"_"+f"{x:05d}"+r'.npy'
                    if not os.path.isfile(filename):
                        save(filename,cp.asnumpy(grey[x]))
                grey[x]=boxreshape(grey[x],desired_shape) #Arrays skalieren
                grey[x]=cp.asnumpy(grey[x]) #von cupy in numpy wandeln
                if save_separate==True: #Wenn Einzeldateien gewünscht, jeweiliges skaliertes Array abspeichern
                    filename=OC_greyboxes_data_path+r'/'+str(DMC)+r'/greybox_'+OCversion+r'_size'+str(resize_size)+r"_"+DMC+r"_"+f"{x:05d}"+r'.npy'
                    if not os.path.isfile(filename):
                        save(filename,grey[x])
            if save_separate==False: #Wenn Gesamtdatei gewünscht, alle skalierten Arrays in einer Datei abspeichern
                filename=OC_greyboxes_data_path+r'/greyboxes_'+OCversion+r'_size'+str(resize_size)+r"_"+DMC+r'.npy'
                if not os.path.isfile(filename):
                    save(filename,grey)
        else: #wenn Skalieren nicht gewünscht
            grey=cp.asnumpy(grey)  #von cupy in numpy wandeln
            if save_separate==True: #wenn Einzeldateien gewünscht
                for x in range(0,len(grey)): #alle unskalierten Arrays abspeichern
                    filename=OC_greyboxes_data_path+r'/'+str(DMC)+r'/greybox_'+OCversion+r'_unsized_'+DMC+r"_"+f"{x:05d}"+r'.npy'
                    if not os.path.isfile(filename):
                        save(filename,grey[x])
            else: #Wenn Gesamtdatei gewünscht, alle unskalierten Arrays in eine Datei speichern
                filename=OC_greyboxes_data_path+r'/greyboxes_'+OCversion+r"_unsized_"+DMC+r'.npy'
                if not os.path.isfile(filename):
                    save(filename,grey)

### Hilfsfunktionen ###

def get_array_file_paths(path,endstring=".npy"):
    #übergibt eine Liste der Pfade aller mit "endstring" endenden Dateien in dem Verzeichnis mit Unterverzeichnis
    #standardmäßig gibt es alle .npy filepaths zurükck
    array_files = [os.path.join(root, name)
                for root, dirs, files in os.walk(path)
                for name in files
                if name.endswith((endstring))]
    return(array_files)

def get_array_file_paths_with_DMC(path,DMC):
    #übergibt eine Liste der Pfade aller mit "endstring" endenden Dateien in dem Verzeichnis mit Unterverzeichnis
    #standardmäßig gibt es alle .npy filepaths zurükck
    array_files = [os.path.join(root, name)
                for root, dirs, files in os.walk(path)
                for name in files
                if name[3:17]==DMC]
    return(array_files)

def get_pandas_file_paths(path,endstring="max.s_43_min.s_03"):
    #übergibt eine Liste der Pfade aller mit "endstring" endenden Dateien in dem Verzeichnis mit Unterverzeichnis 
    #standardmäßig alle pandas Dateien, welche mit den struc größen zwischen 43 und 3 erzeugt wurden
    pandas_files = [os.path.join(root, name)
                for root, dirs, files in os.walk(path)
                for name in files
                if name.endswith((endstring))]
    return(pandas_files)

def create_OC_bb_for_all(raw_array_data_path,pandas_data_path,maxstruc=3,minstruc=3,overwrite=False,output=False,gpu=0,endstring=".npy",OCversion="99"):
    #Aufruffunktion um bounding boxes für gleich mehrere Arrays zu erzeugen
    #Inputs sind die gleichen wie function "create_OC_boundingboxes"
    #Unterschied ist, dass anstatt eines Arrays ein Ordner mit Arrays übergeben wird und ein Ordner mit den zugehörigen Pandas Dateien
    
    #Eignet sich für die parallele Ausführung, wenn unterschiedliche gpus und Datien angegeben werden!
    #Zum angeben unterschiedlicher Dateien können unterschiedliche Ordner angegeben werden oder
    #mit "endstring" kann nach am Ende(!) umbenannten Arrays gefiltert werden, z.b. x.npy

    #overwrite=False bedeutet, dass Arrays übersprungen werden, die mit den angegebenen struc Größen 
    #bereits verarbeitet wurden (und das entsprechende pandasfile im Zielordner gefunden wird)
    #overwrite=True bedeutet, dass etwaige pandas Datien überschrieben werden

    #Erwartete Dauer pro CT-Scan: 6h

    array_list=get_array_file_paths(raw_array_data_path, endstring) #finde die zu verarbeitenden Arrays
    number_files=len(array_list) #Zahl der zu verarbeitenden Arrays
    print(str(number_files)," files found")
    if overwrite==False: #Wenn dieses Array mit diesen strucs schon verarbeitet wurde, wird die Datei übersrungen!
        raw_array_path_length=len(raw_array_data_path) #Pfadlänge auslesen, um DMC "von vorne" auslesen zu können
        copy_array_list=array_list*1 #Array hard copy erzeugen
        k=0 #Zähler für entfernte Arrays
        for i in range(0,number_files):#Für alle Arrays prüfen, ob pandas file existiert
            #print(copy_array_list) #Prüfausgabe
            DMC=copy_array_list[i][raw_array_path_length+4:raw_array_path_length+18] #DMC auslesen
            for dir,sub_dirs, files in os.walk(pandas_data_path): #pandas file suchen
                for name in files: 
                    if name.endswith(str(DMC)+r'_max.s_'+f"{maxstruc:02d}"+r'_min.s_'+f"{minstruc:02d}"): #wenn pandas file gefunden
                        print(str(DMC)+r'_max.s_'+f"{maxstruc:02d}"+r'_min.s_'+f"{minstruc:02d}")
                        array_list.remove(copy_array_list[i]) #Array file aus liste entfernen
                        k=k+1
    print('Number of those already transformed ', k)
    print(str(len(array_list))," files will be processed")
    for i in array_list: #für alle restlichen Arrays
        print("processing file: ", str(i))
        DMC=i[raw_array_path_length+4:raw_array_path_length+18] #Aktuellen DMC auslesen
        print(DMC)
        create_OC_boundingboxes(i,pandas_data_path, DMC=DMC, maxstruc=maxstruc, output=output,minstruc=minstruc,gpu=gpu,OCversion=OCversion) #Bounding boxes erzeugen

def combine_pandas_files(pandas_data_path,endstring="max.s_43_min.s_03"):
    #Erzeugt eine pandas file aus allen pandas files, die in dem Ordner gefunden werden
    #Entfernt Duplikate

    pandas_files = [os.path.join(root, name)
                for root, dirs, files in os.walk(pandas_data_path)
                for name in files
                if name.endswith((endstring))]
    print("number of files found:",len(pandas_files))
    
    #neues pandas dataset erzeugen, indem die erste Datei ausgelesen wird und aus der liste gelöscht wird
    with open(pandas_files.pop(0), "rb") as data0:
        boundingboxes=pickle.load(data0)

    #alle weiteren pandas Dateien anhängen
    for i in pandas_files:
        with open(i, "rb") as bb_data:
            boxes_total=pickle.load(bb_data)
            #print("i length:", str(len(boxes_total)))
            #print("bb length:", str(len(boundingboxes)))
            boundingboxes=pd.concat([boundingboxes,boxes_total],ignore_index=True)
            #print("new bb length:", str(len(boundingboxes)))
            #print("count:\n", boundingboxes["DMC"].value_counts())
    print("total bb length:", str(len(boundingboxes)))
    boundingboxes.drop_duplicates(inplace=True) #Duplikate entfernen
    print("total bb length after removing duplicates:", str(len(boundingboxes)))
    print("final bb length without duplicates:", str(len(boundingboxes)))
    boundingboxes.to_pickle(pandas_data_path+"/pd_bb_v02_combined_max.s_43_min.s_03_neu")

def create_OC_greyboxes_for_all(raw_array_data_path,pandas_single_data_path,OC_greyboxes_data_path,resize_size=64,save_separate=True,gpu=0):
    array_list=get_array_file_paths(raw_array_data_path)
    #print(array_list)
    with open(pandas_single_data_path, "rb") as pandas_file:
        print(pandas_single_data_path)
        bounding_boxes=pickle.load(pandas_file)
        bb_array_list=bounding_boxes["DMC"].unique() #find all DMC codes and save DMC into list to find array
        print(bb_array_list)
        for i in bb_array_list: #for all DMC codes
            raw_array_single_data_path=get_array_file_paths_with_DMC(raw_array_data_path, i)[0] #find Array with DMC
            pandas_data=bounding_boxes.loc[bounding_boxes['DMC'] == i] #extract all bounding boxes from pandas dataset for this DMC
            print(raw_array_single_data_path) #DMC array being processed
            create_greyboxes_with_bb(raw_array_single_data_path,pandas_data,OC_greyboxes_data_path,resize_size=resize_size,save_separate=save_separate,gpu=gpu) #create bounding boxes