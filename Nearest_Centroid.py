# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 08:35:03 2021

@author: Trill
"""
# Nearest class centroid classifier 
# Using scikit-learns implementation: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html
# Visualization inspired by: https://scikit-learn.org/stable/auto_examples/neighbors/plot_nearest_centroid.html#sphx-glr-auto-examples-neighbors-plot-nearest-centroid-py
# Each class is represented by its centroid
# New samples are classified based on nearest class centroid. 
# Data can only be vizualised in 2D. 

class Subclass_Nearest_Centroid:
    def __init__(self, n_c):
        self.n_clusters = n_c 
        
    def fit(self, x_train, y_train):
        import numpy as np
        from sklearn.cluster import KMeans
        # Unsupervised cluster creation based on class segments
        labels = np.unique(y_train)
        x_label_train = [] * len(x_train)
        y_label_train = [] * len(y_train)
        self.k_models = [None] * len(labels)
        
        for i in range(len(labels)):
            indexes = np.where(y_train == i)
            x_label_train.append(x_train[indexes])
            y_label_train.append(y_train[indexes])
            self.k_models[i] = KMeans(self.n_clusters).fit(x_label_train[i])
            
            
    def predict(self, x_test):
        import numpy as np
        m_centroid = [None] * len(self.k_models)
        for idx, m in enumerate(self.k_models):
            m_centroid[idx] = m.cluster_centers_

        # Loop through each x-value 
        # Find eucledian distance for each cluster 
        # Predict x-value with closest cluster model.
        y_pred = np.array([None] * len(x_test))
        
        for idx, x in enumerate(x_test): 
            current_best = float("inf")
            current_best_class = -1
            for k, m_c in enumerate(m_centroid):
                for c in range(self.n_clusters):
                    cluster_distance = np.linalg.norm(np.abs(m_c[c] - x)) 
                    if cluster_distance < current_best:
                        current_best = cluster_distance 
                        current_best_class = k 
            y_pred[idx] = current_best_class
            
        return y_pred
            