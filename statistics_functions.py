import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap



def single_cluster_analysis_plot(pandas_single_file_path,cluster_number,PCA=True,TSNE=False,label_name='NA'):

###Auswertung einzelner Cluster
    
    with open(pandas_single_file_path, "rb") as bb_data:
        pd_data=pd.read_pickle(bb_data)
    #print(pd_data)
    if label_name=='NA':
        label_name=pd_data.filter(regex="label").columns[-1]
    num_clusters=len(pd_data[label_name].unique())

    print(f'Anzahl an Grauboxen: {len(pd_data)}')
    print(f'Anzahl an Clustern: {num_clusters}')
    print(f'Spaltenname der Labels: {label_name}')

    vector=pd_data.filter(regex="feature") #only features
    if PCA==True:
        vector=pd.concat([vector,pd_data.filter(regex="pca")],axis=1)
    if TSNE==True:
        vector=pd.concat([vector,pd_data.filter(regex="tsne")],axis=1)
    vector=vector.apply(lambda x: ((x-x.min())/(x.max()-x.min())) if x.max()!=x.min() else x, axis=0)
    pd_data.update(vector)
    pd_data=pd_data.loc[pd_data[label_name] == cluster_number]
    vector=pd_data.filter(regex="feature") #only features
    if PCA==True:
        vector=pd.concat([vector,pd_data.filter(regex="pca")],axis=1)
    if TSNE==True:
          vector=pd.concat([vector,pd_data.filter(regex="tsne")],axis=1)
    
     #min max skaling
    print(vector)
    print("länge",len(vector))
    print(vector.median())
    fig1, ax1 = plt.subplots(figsize=(15,10))
    ax1.set_title('Featuredistribution of '+label_name[:-6]+' Cluster '+str(cluster_number)+' with '+str(len(vector))+" greyboxes")
    flierprops = dict(marker='x', markerfacecolor='r', markersize=1,
                  linestyle='none')
    ax1.boxplot(vector,flierprops=flierprops,showmeans=True,meanprops={"marker":"^","markerfacecolor":"orange", "markeredgecolor":"orange"})
    ax1.legend(["feature distribution"])
    namelist=[]
    plt.ylim((0,1))
    plt.xticks(range(1,len(vector.columns)+1),vector.columns)
    plt.xticks(rotation=45,fontsize=8)
    mean_legend = mlines.Line2D([], [], color='orange', marker='^', linestyle='None',
                          markersize=6, label='mean value')
    plt.legend(handles=[mean_legend], loc=1, fontsize=8)
    
    plt.tight_layout()
    plt.show()

### Auswertung aller Cluster (3D-Plot)


def cluster_analysis_plot(pandas_single_file_path,PCA=True,TSNE=False,label_name='NA'):

    with open(pandas_single_file_path, "rb") as bb_data:
        pd_data=pd.read_pickle(bb_data)
    
    if label_name=='NA':
        label_name=pd_data.filter(regex="label").columns[-1]
    num_clusters=len(pd_data[label_name].unique())

    print(f'Anzahl an Grauboxen: {len(pd_data)}')
    print(f'Anzahl an Clustern: {num_clusters}')
    print(f'Spaltenname der Labels: {label_name}')

    vector=pd_data.filter(regex="feature") #only features
    if PCA==True:
        vector=pd.concat([vector,pd_data.filter(regex="pca")],axis=1)
    if TSNE==True:
        vector=pd.concat([vector,pd_data.filter(regex="tsne")],axis=1)

    #plot_data=nparray((num_features+num_PCA+num_TSNE)*Anzahl Auswertungen,Anzahl Cluster)
    vector=vector.apply(lambda x: ((x-x.min())/(x.max()-x.min())) if x.max()!=x.min() else x, axis=0)
    pd_data.update(vector)
    plot_data=np.zeros((len(vector.columns),num_clusters))
    #plot_data=np.zeros((len(vector.columns),10))
    for i in range(0,num_clusters):
        cluster_pd_data=pd_data.loc[pd_data[label_name] == i]
        vector=cluster_pd_data.filter(regex="feature") #only features
        if PCA==True:
            vector=pd.concat([vector,cluster_pd_data.filter(regex="pca")],axis=1)
        if TSNE==True:
            vector=pd.concat([vector,cluster_pd_data.filter(regex="tsne")],axis=1)    
        plot_data[:,i]=vector.mean()
        if i % 10 == 0:
            print("cluster "+str(i)+" finished")
    print(plot_data)
    colormaps = [ListedColormap(["darkorange", "gold", "lawngreen", "lightseagreen"])]
    viridis_big = cm.get_cmap('rainbow', 20)
    colormaps = [ListedColormap(viridis_big(np.linspace(0, 1, 256)))]
    n = len(colormaps)
    fig, axs = plt.subplots(1, n, figsize=(n * 2 + 2, 3),
                            constrained_layout=True, squeeze=False)
    for [ax, cmap] in zip(axs.flat, colormaps):
        psm = ax.pcolormesh(plot_data, cmap=cmap, rasterized=True, vmin=0, vmax=1)
        fig.colorbar(psm, ax=ax)
    plt.yticks(range(1,len(vector.columns)+1),vector.columns)
    plt.yticks(rotation=45,fontsize=8)
    plt.title('Featuredistribution of all '+label_name[:-6]+' Cluster with a total of '+str(len(pd_data))+" greyboxes")
    plt.show()



def cluster_analysis_feature_dominance(pandas_single_file_path,PCA=True,TSNE=False,label_name='NA'):
    with open(pandas_single_file_path, "rb") as bb_data:
        pd_data=pd.read_pickle(bb_data)

    if label_name=='NA':
        label_name=pd_data.filter(regex="label").columns[-1]
    num_clusters=len(pd_data[label_name].unique())

    print(f'Anzahl an Grauboxen: {len(pd_data)}')
    print(f'Anzahl an Clustern: {num_clusters}')
    print(f'Spaltenname der Labels: {label_name}')

    vector=pd_data.filter(regex="feature") #only features
    if PCA==True:
        vector=pd.concat([vector,pd_data.filter(regex="pca")],axis=1)
    if TSNE==True:
        vector=pd.concat([vector,pd_data.filter(regex="tsne")],axis=1)

    #plot_data=nparray((num_features+num_PCA+num_TSNE)*Anzahl Auswertungen,Anzahl Cluster)
    vector=vector.apply(lambda x: ((x-x.min())/(x.max()-x.min())) if x.max()!=x.min() else x, axis=0)
    pd_data.update(vector)

    plot_data=np.zeros((1,num_clusters))
    #plot_data=np.zeros((len(vector.columns),10))
    for i in range(0,num_clusters):
        cluster_pd_data=pd_data.loc[pd_data[label_name] == i]

        vector=cluster_pd_data.filter(regex="feature") #only features
        if PCA==True:
            vector=pd.concat([vector,cluster_pd_data.filter(regex="pca")],axis=1)
        if TSNE==True:
            vector=pd.concat([vector,cluster_pd_data.filter(regex="tsne")],axis=1)    
        mean_list=vector.median()

        plot_data[0,i]=mean_list.nlargest(n=2).iloc[0]/mean_list.nlargest(n=2).iloc[1]
        if i % 10 == 0:
            print("cluster "+str(i)+" finished")
    #print(plot_data)
    colormaps = [ListedColormap(["darkorange", "gold", "lawngreen", "lightseagreen"])]
    viridis_big = cm.get_cmap('rainbow', 20)
    colormaps = [ListedColormap(viridis_big(np.linspace(0, 1, 8)))]
    n = len(colormaps)
    fig, axs = plt.subplots(1, n, figsize=(n * 2 + 2, 3),
                            constrained_layout=True, squeeze=False)
    for [ax, cmap] in zip(axs.flat, colormaps):
        psm = ax.pcolormesh(plot_data, cmap=cmap, rasterized=True, vmin=0,vmax=8)
        fig.colorbar(psm, ax=ax)
    plt.yticks((0,1))
    #plt.yticks(rotation=45,fontsize=8)
    plt.title('Featuredistribution of all '+label_name[:-6]+' Cluster with a total of '+str(len(pd_data))+" greyboxes")
    plt.show()





directory_path = 'D:/AnomalieKI' #working directory
greyboxes_data_path = directory_path+'/data/04_greyboxes_npy' #where the greyboxes are located
pandas_data_path = directory_path+'/Data/06_pandas_dataset_with_features' #where pandas with boundingboxes and features is located
pandas_single_file_path=pandas_data_path+r'/pd_bb_v02_combined_max.s_43_min.s_03_with_16_features_v7r6'# neues
pandas_data_path_labels=directory_path+'/Data/07_pandas_dataset_with_labels'
pandas_single_file_path_labels=pandas_data_path_labels+r'/pd_bb_v02_combined_max.s_43_min.s_03_with_16_features_v7r6_labels_kmeans_300'# neues

#cluster_analysis_plot(pandas_single_file_path_labels)
#cluster_analysis_feature_dominance(pandas_single_file_path_labels)
#single_cluster_analysis_plot(pandas_single_file_path_labels,cluster_number=47,PCA=False,TSNE=False)

#def Maximalauswertung Features
    
    #open pands
    #erstelle neues pandas dataframe (Clusternummer, Max-Featurename, Max-Wert, Verhältnis)
    #für alle unique label
        #über alle features finde max
        #über alle features außer feature mit max, finde max
        #teile erstes max durch zweites max
        #speichere clusternummer, feature name des ersten max  und Verhältnis von ersten zu zweiten ab, sowie ersten und zweiten Wert als neue Zeile in neues Pandas df

    
