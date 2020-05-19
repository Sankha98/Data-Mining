#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 1 19:22:38 2020

@author: sankha
"""

'''
Compute the following cluster validation indices:
Silhouette index, Dunne Index, Davies Bouldin Index
'''

#   Import Libraries 
import math 
import random
from urllib.request import urlretrieve
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import normalize,StandardScaler

# K-Means
def KMeansClustering(X, n_cluster):
    model = KMeans(n_clusters = n_cluster)
    model.fit(X)
    labels = model.predict(X)
    return labels
  
def getSilhoutteIndex(clusters,arr = []):
    index = 0
    total = 0
    for cluster_key in clusters:
        this_cluster = clusters[cluster_key]
        this_cluster_length = len(this_cluster)
        total = total + this_cluster_length
        for ele in this_cluster:
            sum = 0
            #Calculate two points a and b i.e. coesion and seperation
            for other_ele in this_cluster:
                # get euclidean dist here between ele and other_ele
                dist = 0
                for i in range(len(ele)):
                    if i not in arr:
                        dist = dist + (ele[i] - other_ele[i]) ** 2
                dist = math.sqrt(dist)
                sum = sum + dist
            if (this_cluster_length == 1):
                a = 0
            else:
                a = sum / (this_cluster_length - 1)
            # Getting value of b
            b = 999999999
            for other_cluster_key in clusters:
                if (other_cluster_key == cluster_key):
                    continue
                other_cluster = clusters[other_cluster_key]
                other_cluster_length = len(other_cluster)
                sum1 = 0
                for other_cluster_ele in other_cluster:
                    dist = 0
                    for i in range(len(ele)):
                        if i not in arr:
                            dist = dist + (ele[i] - other_cluster_ele[i]) ** 2
                    dist = math.sqrt(dist)
                    sum1 = sum1 + dist
                b = min(b,sum1/other_cluster_length)
            cluster_index = 0
            if (a != b):
                #storing the Silhoutte coefficient value
                cluster_index = ((b-a)/max(a,b))
            index = index + cluster_index
    index = index / total
    return index

#   DaviesBouldin Index
def getDBIndex(X,labels):
    return davies_bouldin_score(X,labels)

#    Helper function for containing attributes of that each np arrays

def getCentroid(cluster):
    length = len(cluster)
    centroid = []
    first =  True
    for ele in cluster:
        if first:
            first = False
            for i in range(len(ele)):
                if np.isreal(ele[i]):
                    centroid.append(ele[i])
                else:
                    centroid.append(0)
            continue
        for i in range(len(ele)):
            if np.isreal(ele[i]):
                centroid[i] = centroid[i]  + ele[i]
            else:
                centroid[i] = centroid[i] + 0
    centroid = np.asarray(centroid)
    centroid = centroid / length
    return centroid

# Dunn Index
def getDunnIndex(clusters,arr=[]):
    cluster_in_between = 9999999
    for cluster_key in clusters:
        examp_temp = getCentroid(clusters[cluster_key])
        for cluster_key1 in clusters:
            if cluster_key == cluster_key1:
                continue
            examp_temp1 = getCentroid(clusters[cluster_key1])
            dist = 0
            for i in range(len(examp_temp)):
                if i not in arr:
                    dist+=(examp_temp[i]-examp_temp1[i])**2
            dist = math.sqrt(dist)
            if (cluster_in_between > dist):
                cluster_in_between = dist
    max_intra_cluster = 0
    for cluster_key in clusters:
        max_intra_cluster = 0
        this_cluster = clusters[cluster_key]
        for data in this_cluster:
            for data1 in this_cluster:
                dist = 0
                for i in range(len(data)):
                    if i not in arr:
                        dist+=(data[i]-data1[i])**2
                dist = math.sqrt(dist)
                if (dist > max_intra_cluster):
                    max_intra_cluster = dist
        if (max_intra_cluster > max_intra_cluster):
            max_intra_cluster = max_intra_cluster
    return (cluster_in_between/max_intra_cluster)

#   Main
df = pd.read_csv("iris.csv")
scaler = StandardScaler()
X = np.array(scaler.fit_transform(df.drop(["class"],1).astype(float)))
# plt.plot(list(range(1,X.shape[0]+1)), distanceDec)
# plt.show()
labels = KMeansClustering(X,3)

df1 = pd.read_csv("wine.csv")
scaler1 = StandardScaler()
#scaler1 = MinMaxScaler()
X1 = np.array(scaler1.fit_transform(df1.drop(["class"],1).astype(float)))
labels1 = KMeansClustering(X1,2)

df2 = pd.read_csv("wdbc.csv")
scaler2 = StandardScaler()
#scaler2 = MinMaxScaler() 
X2 = np.array(scaler2.fit_transform(df2.drop(['class'],1).astype(float)))
labels2 = KMeansClustering(X2,3)

#   Dictionary with key as cluster number and,value as a list of points in that cluster (each point is a np array of attributes)
clusters={}
k = 0
for i in labels:
    if i in clusters:
        clusters[i].append(df.iloc[k].values)
    else:
        clusters[i] = [df.iloc[k].values]
    k = k + 1

clusters1={}
x = 0
for i in labels1:
    if i in clusters1:
        clusters1[i].append(df1.iloc[x].values)
    else:
        clusters1[i] = [df1.iloc[x].values]
    x = x + 1

clusters2={}
y = 0
for i in labels2:
    if i in clusters2:
        clusters2[i].append(df2.iloc[y].values)
    else:
        clusters2[i] = [df2.iloc[y].values]
    y = y + 1

# printing results
print("Iris:")
for item in clusters:
    print()
    print("Cluster ",item)
    print("Length: ",len(clusters[item]))
    
print()
print("BCW:")
for item in clusters1:
    print()
    print("Cluster ",item)
    print("Length: ",len(clusters1[item]))

print()
print("Seeds:")
for item in clusters2:
    print()
    print("Cluster ",item)
    print("Length: ",len(clusters2[item]))
print()
'''
del clusters[-1]
to_remove = []
for i in range(len(labels)):
    if labels[i] == -1:
        to_remove.append(i)
X = np.delete(X,to_remove,0)
labels = np.delete(labels,to_remove)

del clusters1[-1]
to_remove1 = []
for i in range(len(labels1)):
    if labels1[i] == -1:
        to_remove1.append(i) 
X1 = np.delete(X1,to_remove1,0)
labels1 = np.delete(labels1,to_remove1)

del clusters2[-1]
to_remove2 = []
for i in range(len(labels2)):
    if labels2[i] == -1:
        to_remove2.append(i)
X2 = np.delete(X2,to_remove2,0)
labels2 = np.delete(labels2,to_remove2)
'''
# printing the indices
print("Different types of Index vales")
print("Silhoutte Index :",getSilhoutteIndex(clusters,[4]))
print("Davies Bouldin Index :",getDBIndex(X,labels))
print("Dunn Index :",getDunnIndex(clusters,[4]))

print()
print("Silhoutte Index :",getSilhoutteIndex(clusters1,[0]))
print("Davies Bouldin Index :",getDBIndex(X1,labels1))
print("Dunn Index :",getDunnIndex(clusters1,[0]))

print()
print("Silhoutte Index :",getSilhoutteIndex(clusters2,[0]))
print("Davies Bouldin Index :",getDBIndex(X2,labels2))
print("Dunn Index :",getDunnIndex(clusters2,[0]))