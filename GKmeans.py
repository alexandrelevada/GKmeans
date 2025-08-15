#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

A graph-based k-means algorithm for clustering high dimensional data

Clustering high-dimensional data remains a fundamental yet challenging task in unsupervised learning, 
where the effectiveness of classical algorithms like k-means is hindered by the curse of dimensionality 
and their reliance on Euclidean distances. In this work, we introduce GK-means, a novel graph-based 
extension of the k-means algorithm that leverages geodesic distances computed over a k-nearest neighbors 
graph (k-NNG) to better reflect the intrinsic geometry of data manifolds. By replacing Euclidean distances
with geodesic distances obtained via Dijkstra’s algorithm, the proposed method can detect clusters of 
arbitrary shapes and densities, offering improved robustness to noise and high-dimensional sparsity. 
Unlike many state-of-the-art clustering approaches, our algorithm maintains linear complexity in the 
number of samples and edges, making it scalable to large datasets. Extensive experiments on several 
real-world high-dimensional datasets demonstrate that GK-means outperforms both classical k-means 
and the non-optimized HDBSCAN algorithm (default hyperparameters) across multiple cluster quality 
metrics. These results establish GK-means as a powerful and computationally efficient alternative 
for structure discovery in complex high dimensional data.


"""

# Imports
import sys
import time
import warnings
import networkx as nx
import sklearn.datasets as skdata
import matplotlib.pyplot as plt
import numpy as np
import sklearn.neighbors as sknn
import sklearn.utils.graph as sksp
import scipy.sparse._csr
from numpy import dot
from numpy import trace
from numpy.linalg import inv
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import fowlkes_mallows_score


# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

# Geodesic K-means implementation
def GKmeans(dados, nn):
    print('Running GKmeans...')
    iteracao = 0
    # Number of samples, features and classes
    n = dados.shape[0]
    m = dados.shape[1]
    k = len(np.unique(target))
    # Initial random centers
    coord_centros = np.random.choice(n, size=k, replace=False)    
    centros = dados[coord_centros]
    # Add centers at the end of data matrix (k last rows)
    dados = np.vstack((dados, centros))
    # Main loop
    while True:
        iteracao += 1          
        # Create the k-NNG
        knnGraph = sknn.kneighbors_graph(dados, n_neighbors=nn, mode='distance')
        # Adjacency matrix
        W = knnGraph.toarray()
        # NetworkX format
        G = nx.from_numpy_array(W)
        # Array for geodesic distances
        distancias = np.zeros((k, n+k))
        for j in range(k):
            # Dijkstra's algorithm for geodesic distances
            length, path = nx.single_source_dijkstra(G, coord_centros[j])
            # Sort vertices
            dists = list(dict(sorted(length.items())).values()) 
            distancias[j, :] = dists
        # Labels vector
        rotulos = np.zeros(n)    
        # Assign labels to data points
        for j in range(n):
            rotulos[j] = distancias[:, j].argmin()
        # Find the points belonging to each partition
        novos_centros = np.zeros((k, m))
        for r in range(k):
            indices = np.where(rotulos==r)[0]
            if len(indices) > 0:
                sample = dados[indices]
                novos_centros[r, :] = sample.mean(axis=0)
            else:
                novos_centros[r, :] = centros[r, :]
        # Update the last k rows of data (centroids)
        dados[n:n+k, :] = novos_centros
        # Check for convergence
        if (np.linalg.norm(centros - novos_centros) < 0.5) or iteracao > 20:
            break
        # Update the centers
        centros = novos_centros.copy()   
    return rotulos

#%%%%%%%%%%%%%%%%%%%%  Data loading
##### First set of experiments
X = skdata.fetch_openml(name='AP_Colon_Kidney', version=1)         
#X = skdata.fetch_openml(name='AP_Breast_Kidney', version=1)
#X = skdata.fetch_openml(name='AP_Colon_Prostate', version=1)      
#X = skdata.fetch_openml(name='AP_Prostate_Kidney', version=1)         
#X = skdata.fetch_openml(name='AP_Uterus_Kidney', version=1)
#X = skdata.fetch_openml(name='AP_Prostate_Uterus', version=1)
#X = skdata.fetch_openml(name='AP_Lung_Kidney', version=1)
#X = skdata.fetch_openml(name='AP_Ovary_Kidney', version=1)
#X = skdata.fetch_openml(name='AP_Ovary_Lung', version=1)
#X = skdata.fetch_openml(name='AP_Prostate_Ovary', version=1)
#X = skdata.fetch_openml(name='AP_Prostate_Uterus', version=1)
#X = skdata.fetch_openml(name='AP_Omentum_Prostate', version=1)
#X = skdata.fetch_openml(name='tr11.wc', version=1)
#X = skdata.fetch_openml(name='tr45.wc', version=1)
#X = skdata.fetch_openml(name='tr31.wc', version=1)

###### Second set of experiments: HDBSCAN x GK-means
#X = skdata.load_digits()
#X = skdata.fetch_openml(name='11_Tumors', version=1)
#X = skdata.fetch_openml(name='UMIST_Faces_Cropped', version=1)    
#X = skdata.fetch_openml(name='Olivetti_Faces', version=1)
#X = skdata.fetch_openml(name='BurkittLymphoma', version=1)    
#X = skdata.fetch_openml(name='variousCancers_final', version=1)
#X = skdata.fetch_openml(name='mfeat-fourier', version=1)
#X = skdata.fetch_openml(name='mfeat-karhunen', version=1)
#X = skdata.fetch_openml(name='mfeat-factors', version=1)
#X = skdata.fetch_openml(name='mfeat-pixel', version=1)
#X = skdata.fetch_openml(name='mfeat-zernike', version=1)
#X = skdata.fetch_openml(name='optdigits', version=1)
#X = skdata.fetch_openml(name='cnae-9', version=1)
#X = skdata.fetch_openml(name='satimage', version=1)
#X = skdata.fetch_openml(name='semeion', version=1)
#X = skdata.fetch_openml(name='vehicle', version=1)
#X = skdata.fetch_openml(name='micro-mass', version=1)
#X = skdata.fetch_openml(name='JapaneseVowels', version=1)
#X = skdata.fetch_openml(name='waveform-5000', version=1)
#X = skdata.fetch_openml(name='audiology', version=1)
#X = skdata.fetch_openml(name='wine-quality-white', version=1)
#X = skdata.fetch_openml(name='nursery', version=1)
#X = skdata.fetch_openml(name='led24', version=1)
#X = skdata.fetch_openml(name='one-hundred-plants-shape', version=1) 
#X = skdata.fetch_openml(name='one-hundred-plants-texture', version=1)

dados = X['data']
target = X['target']  

n = dados.shape[0]
m = dados.shape[1]
c = len(np.unique(target))

#nn = round(np.sqrt(n))
nn = round(np.log2(n))

print('N = ', n)
print('M = ', m)
print('C = ', c)
print('K = ', nn)
print()

# Sparse matrix (for some high dimensional datasets)
if type(dados) == scipy.sparse._csr.csr_matrix:
    dados = dados.todense()
    dados = np.asarray(dados)
else:
    if not isinstance(dados, np.ndarray):
        cat_cols = dados.select_dtypes(['category']).columns
        dados[cat_cols] = dados[cat_cols].apply(lambda x: x.cat.codes)
        # Convert to numpy
        dados = dados.to_numpy()
        target = target.to_numpy()

# Remove nan's
dados = np.nan_to_num(dados)

# Convert labels to integers
rotulos = list(np.unique(target))
numbers = np.zeros(n)
for i in range(n):
    numbers[i] = rotulos.index(target[i])
target = numbers

#########################
# Execution of GKmeans
#########################
MAX = 51
inicio = time.time()
lista_rand = []
lista_v = []
lista_fm = []
for i in range(1, MAX):
    labels = GKmeans(dados, nn)
    # External indices
    lista_rand.append(rand_score(target, labels))
    lista_v.append(v_measure_score(target, labels))
    lista_fm.append(fowlkes_mallows_score(target, labels))
fim = time.time()

print()
print('GEODESIC K-MEANS')
print('-----------------')
print('Elapsed time: %.4f' %(fim - inicio))
print()
print('Average Rand index: %.4f' %(sum(lista_rand)/(MAX-1)))
print('Average V-measure: %.4f' %(sum(lista_v)/(MAX-1)))
print('Average FM: %.4f' %(sum(lista_fm)/(MAX-1)))

# ######################
# # Execution of kmeans
# ######################
inicio = time.time()
lista_rand = []
lista_v = []
lista_fm = []
for i in range(1, MAX):
    kmeans = KMeans(n_clusters=c, init='random', n_init=1).fit(dados)
    # External indices
    lista_rand.append(rand_score(target, kmeans.labels_))
    lista_v.append(v_measure_score(target, kmeans.labels_))
    lista_fm.append(fowlkes_mallows_score(target, kmeans.labels_))
fim = time.time()

print()
print('K-MEANS')
print('----------')
print('Elapsed time: %.4f' %(fim - inicio))
print()
print('Average Rand index: %.4f' %(sum(lista_rand)/(MAX-1)))
print('Average V-measure: %.4f' %(sum(lista_v)/(MAX-1)))
print('Average FM: %.4f' %(sum(lista_fm)/(MAX-1)))

########################
# Execution of HDBSCAN 
########################
inicio = time.time()
hdbscan = HDBSCAN(min_cluster_size=10).fit(dados)
fim = time.time()
print()
print('HDBSCAN')
print('----------')
print('Elapsed time: %.4f' %(fim - inicio))
print()
print('Rand index: %.4f' %(rand_score(target, hdbscan.labels_)))
print('V-measure: %.4f' %(v_measure_score(target, hdbscan.labels_)))
print('FM: %.4f' %(fowlkes_mallows_score(target, hdbscan.labels_)))

########################
# Execution of GMM 
########################
inicio = time.time()
GMM_labels = GaussianMixture(n_components=c, reg_covar=0.0001, random_state=42).fit_predict(dados)
fim = time.time()
print()
print('GMM')
print('-----')
print('Elapsed time: %.4f' %(fim - inicio))
print()
print('Rand index: %.4f' %(rand_score(target, GMM_labels)))
print('V-measure: %.4f' %(v_measure_score(target, GMM_labels)))
print('FM: %.4f' %(fowlkes_mallows_score(target, GMM_labels)))

#########################################
# Execution of Agglomerative clustering 
#########################################
inicio = time.time()
clustering = AgglomerativeClustering(n_clusters=c, linkage='complete').fit(dados)
fim = time.time()
print()
print('Agglomerative Clustering')
print('-------------------------')
print('Elapsed time: %.4f' %(fim - inicio))
print()
print('Rand index: %.4f' %(rand_score(target, clustering.labels_)))
print('V-measure: %.4f' %(v_measure_score(target, clustering.labels_)))
print('FM: %.4f' %(fowlkes_mallows_score(target, clustering.labels_)))