# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 15:48:46 2025

@author: LAB
"""

import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pickle

#load model
with open('kmeans_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
    
#set page config
st.set_page_config(page_title = "K-means Clustering", layout = "centered")
    
#set title application
st.title("📈 K-means clustering Visualizer by Pornprom Ounsuchat")

#load dataset
X, _ =  make_blobs(n_samples=300, centers=loaded_model.n_clusters, cluster_std=0.60, random_state=0)

#Predict using loaded model
y_kmeans = loaded_model.predict(X)

#plotting
fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
ax.scatter(loaded_model.cluster_centers_[:, 0], loaded_model.cluster_centers_[:, 1], s=300, c='red')
ax.set_title('k-Means Clustering')
ax.legend()
st.pyplot(fig)
  
