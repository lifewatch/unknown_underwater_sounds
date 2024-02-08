# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 08:40:33 2023

Grid search function defined to tune the hyperparameters of sparce PCA and HDBSCAN. 
sPCA applied to both AVES and AE (or CAE) feature sets

@author: arienne.calonge
"""

import os
import pandas as pd 
import hdbscan
import umap.umap_ as umap
import umap.plot
from sklearn.preprocessing import scale
from sklearn import metrics
from sklearn.decomposition import SparsePCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AffinityPropagation
import dbcv #https://github.com/FelSiq/DBCV
import numpy as np

def complete_grid_search(df0, df_labels, parameters, results):    
    if feature_extraction == "AE": 
        for r in parameters['alpha']:     
            spca = SparsePCA(n_components=3, random_state=5, alpha = r)
            data_spca = spca.fit(df0)
            components = data_spca.components_
            x = pd.DataFrame(components, index=np.array(range(0, 3)), columns=np.array(range(0, df0.shape[1]))).T     
            x['sum'] = x.abs().sum(axis=1)
            
            #get list of features to be included in the model
            x = x.drop(x[x['sum'] == 0].index).T
            features = list(x.columns)
            
            #prepare dataframe
            df = df0[features]
            df = pd.concat([df,df0_imp_features], axis =1)   
            df.columns = df.columns.astype(str)
            
            n_samples = df.shape[0] 
            n_features = df.shape[1] 
            
            #scale 
            df = pd.DataFrame(scale(df), index=df.index, columns=df.columns)
            
            for s in parameters['min_cluster_size']:
                for t in parameters['min_samples']:
                    for u in parameters['epsilon']:
                        data_hdbscan = hdbscan.HDBSCAN(min_cluster_size=s, min_samples=t, cluster_selection_epsilon=u,cluster_selection_method="leaf").fit(df)
                        cluster = pd.DataFrame(data_hdbscan.labels_)
                        cluster.columns = ["cluster"]
                        df_hdbscan = pd.concat([df, df_labels, cluster], axis=1)
                        df_hdbscan = df_hdbscan.drop_duplicates() #AE
                        noise_count = df_hdbscan['cluster'].value_counts()[-1]
                        df_hdbscan = df_hdbscan.loc[df_hdbscan['cluster'] > -1] 
            
                        df_hdbscan.to_csv(dir_grid_search+str(len(results))+'_results_grid_search.csv')
            
                        percentage_samples = len(df_hdbscan)/len(df)
                                
                        #calculate homogeneity
                        label = df_hdbscan['label']
                        cluster = df_hdbscan['cluster']
                        homogeneity = metrics.homogeneity_score(label, cluster)
                                        
                        #calculate DBCV
                        n = len(df_hdbscan.columns)-2
                        df_hdbscan_features = df_hdbscan.iloc[:,1:n]                    
                        dbcv_score = dbcv.dbcv(df_hdbscan_features, cluster)
                        
                        #append results to dataframe
                        no_clusters = df_hdbscan['cluster'].nunique()
                        new_row = {"Number of features": n_features, 'epsilon':u,"Samples": n_samples,
                                   "sPCA alpha": r, "sPCA selected features": features,
                                   "no_clusters": no_clusters, "min_cluster_size": s, 
                                   "min_samples": t, "no_clusters": no_clusters,
                                   "homogeneity": homogeneity, "DBCV": dbcv_score,
                                   "noise_count": noise_count, 'percentage_samples':percentage_samples}
                        results.loc[len(results)] = new_row
                        print("sPCA alpha, epsilon, min_cluster_size, min_samples:", r, u, s, t)
            print("AE Grid search complete")
    elif feature_extraction == "Aves":
        for r in parameters['alpha']:     
            spca = SparsePCA(n_components=3, random_state=5, alpha = r)
            data_spca = spca.fit(df0)
            components = data_spca.components_
            x = pd.DataFrame(components, index=np.array(range(0, 3)), columns=np.array(range(0, df0.shape[1]))).T     
            x['sum'] = x.abs().sum(axis=1)
            
            #get list of features to be included in the model
            x = x.drop(x[x['sum'] == 0].index).T
            features = list(x.columns)
            
            #prepare dataframe
            df = df0[features]
            df = pd.concat([df,df0_imp_features], axis =1)   
            df.columns = df.columns.astype(str)
            
            n_samples = df.shape[0] 
            n_features = df.shape[1] 
            
            #scale 
            df = pd.DataFrame(scale(df), index=df.index, columns=df.columns)
        
            for s in parameters['min_cluster_size']:
                    for t in parameters['min_samples']:
                        for u in parameters['epsilon']:
                            data_hdbscan = hdbscan.HDBSCAN(min_cluster_size=s, min_samples=t, cluster_selection_epsilon=u,cluster_selection_method="leaf").fit(df)
                            cluster = pd.DataFrame(data_hdbscan.labels_)
                            cluster.columns = ["cluster"]
                            df_hdbscan = pd.concat([df, df_labels, cluster], axis=1)
                            noise_count = df_hdbscan['cluster'].value_counts()[-1]
                            df_hdbscan = df_hdbscan.loc[df_hdbscan['cluster'] > -1] 
                            df_hdbscan.to_csv("C:/Users/arienne.calonge/Py/Acoustics_paper/results_grid_search_2.1/"+str(len(results)-1)+'_results_grid_search.csv')
                
                            percentage_samples = len(df_hdbscan)/len(df0)
                                    
                            #calculate homogeneity
                            label = df_hdbscan['label']
                            cluster = df_hdbscan['cluster']
                            homogeneity = metrics.homogeneity_score(label, cluster)
                                            
                            #calculate DBCV
                            n = len(df_hdbscan.columns)-2
                            df_hdbscan_features = df_hdbscan.iloc[:,1:n]                    
                            dbcv_score = dbcv.dbcv(df_hdbscan_features, cluster)
                            
                            #append results to dataframe
                            no_clusters = df_hdbscan['cluster'].nunique()
                            new_row = {"Number of features": n_features, 'epsilon':u,"Samples": n_samples,
                                       "sPCA alpha": r, "sPCA selected features": features,
                                       "no_clusters": no_clusters, "min_cluster_size": s, 
                                       "min_samples": t, "no_clusters": no_clusters,
                                       "homogeneity": homogeneity, "DBCV": dbcv_score,
                                       "noise_count": noise_count, 'percentage_samples':percentage_samples}
                            results.loc[len(results)] = new_row
                            print("sPCA alpha, epsilon, min_cluster_size, min_samples:", r, u, s, t)
        print("AVES grid search complete")
   
def plot_heatmap(labels, predictions):
    results = pd.DataFrame({'label': labels, 'cluster': predictions})

    clusters_numbers = results['cluster'].unique()
    clusters_numbers.sort()
    heatmap = pd.DataFrame(index=results['label'].unique(), columns=clusters_numbers)
    for l, l_df in results.groupby('label'):
        counts = l_df['cluster'].value_counts()
        heatmap.loc[l, counts.index] = counts.values 

    heatmap = heatmap.fillna(0)
    sns.heatmap(heatmap)
    plt.show()
    
def plot_umap(clust_data, clust_cluster):
    prj = umap.UMAP().fit(clust_data)
    umap.plot.points(prj, labels = clust_cluster)

    #PLOT
    umap_embedding = umap.UMAP(n_components=2, random_state=42).fit_transform(clust_data)
    sns.scatterplot(x=umap_embedding[:, 0], y=umap_embedding[:, 1], hue=clust_cluster.astype('str'), s=1)
    plt.show()

         
parameters_AVES_mean = {'alpha':[15, 18, 20],'min_cluster_size':[5,8,10,12], 'min_samples':[3,4,5], 'epsilon':[0.2, 0.5, 0.8]} 
parameters_AVES_max = {'min_cluster_size':[5,8,10,12], 'min_samples':[3,4,5], 'alpha':[35, 45, 55], 'epsilon':[0.2, 0.5, 0.8]} 
parameters_AE_cropped = {'alpha':{8, 10, 12},'min_cluster_size':[5,8,10,12], 'min_samples':[3,4,5], 'epsilon':[0.2, 0.5, 0.8]} 
parameters_AE_cropped_duration = {'alpha':{5,6,7},'min_cluster_size':[5,8,10,12], 'min_samples':[3,4,5], 'epsilon':[0.2, 0.5, 0.8]} 
parameters_AE_standard = {'alpha':{3,4,5},'min_cluster_size':[5,8,10,12], 'min_samples':[3,4,5], 'epsilon':[0.2, 0.5, 0.8]} 

#create empty dataframe
results = pd.DataFrame(columns=['Feature extraction','Feature description', 'sPCA alpha','sPCA selected features',
                                'Number of features',
                                'epsilon','min_cluster_size', 'min_samples', 'no_clusters', 
                                'noise_count','Samples','percentage_samples',
                                'homogeneity', 'DBCV'])

n_start = 0

directory = "C:/Users/arienne.calonge/Py/Acoustics_paper/AE_AVES_2/" #files should be named: AE_feature_cropped, AE_feature_standard, Aves_feature_max, Aves_feature_mean

dir_raw_data = directory+"datasets/"
dir_grid_search = directory+"results_grid_search_2.1/"

#parameters = {'alpha':[0.01],'min_cluster_size':[5], 'min_samples':[5], 'epsilon':[0.1]} 

#AVES features 

for file in os.listdir(dir_raw_data):
    raw = os.path.join(dir_raw_data, file)
    raw_df = pd.read_csv(raw)
    
    #file codes
    feature_extraction = file[file.find('A') : file.find('_f')]
    feature_desc = file[file.find('e_')+2 : file.find('.csv')]
    
    #choose parameters
    if feature_desc == "mean":
        parameters = parameters_AVES_mean
    elif feature_desc == "max":
        parameters = parameters_AVES_max
    elif feature_desc == "cropped":
        parameters = parameters_AE_cropped
    elif feature_desc == "standard":
        parameters = parameters_AE_standard
    elif feature_desc == "cropped_duration":
        parameters = parameters_AE_cropped_duration
        
    #apply filters
    raw_df = raw_df.drop(raw_df[(raw_df['duration'] < 0.02) & (raw_df['duration'] > 10)].index)
    raw_df = raw_df.drop(raw_df[(raw_df['min_freq'] >= 24000)].index)
    raw_df = raw_df.drop(raw_df[(raw_df['max_freq'] > 24000)].index)
    raw_df = raw_df.drop(raw_df[(raw_df['snr'] < 10)].index)
    raw_df = raw_df.drop(['snr'], axis=1) #drop snr
        
    raw_df.dropna(subset=['label'], inplace=True)   
    
    df0 = raw_df.drop(['min_freq','max_freq','duration', 'bandwidth','label'], axis=1).iloc[:,1:] #exclude the 4 important features
    df0_imp_features = raw_df.loc[:, ['min_freq','max_freq','duration', 'bandwidth']] 
    df0.columns = range(df0.columns.size)
    
    df_labels = raw_df["label"]
    
    complete_grid_search(df0, df_labels, parameters, results)
      
    n_end = len(results)-1
    
    results.loc[n_start:n_end,'Feature extraction'] = feature_extraction
    results.loc[n_start:n_end, 'Feature description'] = feature_desc
    
    n_start = n_end + 1
    
results.to_csv("C:/Users/arienne.calonge/Py/Acoustics_paper/AE_AVES_2/results_grid_search_2.1/results_grid_search.csv")

#plot best result from grid search
df_hdbscan = pd.read_csv("C:/Users/arienne.calonge/Py/Acoustics_paper/AE_AVES_2/results_grid_search_2.1/231_results_grid_search.csv")
label = df_hdbscan['label']
cluster = df_hdbscan['cluster']
plot_heatmap(label, cluster)

clust_data = df_hdbscan.iloc[:,0:19]
plot_umap(clust_data, df_hdbscan['cluster'])




