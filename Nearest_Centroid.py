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
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
class Subclass_Nearest_Centroid:
    def __init__(self, n_clusters = 2):
        self.n_clusters = n_clusters 
        
    def fit(self, x_train, y_train):
        # Unsupervised cluster creation based on class segments
        labels = np.unique(y_train)
        x_label_train = [] * len(x_train)
        y_label_train = [] * len(y_train)
        self.k_models = [None] * len(labels)
        
        for idx, lab in enumerate(labels):
            indexes = np.where(y_train == lab)
            x_label_train.append(x_train[indexes])
            y_label_train.append(y_train[indexes])
            self.k_models[idx] = KMeans(self.n_clusters).fit(x_label_train[idx])
            
            
    def predict(self, x_test):
        m_centroid = [None] * len(self.k_models)
        for idx, m in enumerate(self.k_models):
            m_centroid[idx] = m.cluster_centers_

        # Loop through each x-value 
        # Find eucledian distance for each cluster 
        # Predict x-value with closest cluster model.
        y_pred = np.array([0] * len(x_test), dtype=int)
        
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
            
    def grid_search_kfold(self, x_train, y_train, parameters):
        kf = StratifiedKFold(n_splits=5)
        best_score = -1
        best_parameter = -1
        
        for value in parameters:
            for train_index, test_index in kf.split(x_train, y_train):
                x_cv_train, x_cv_test = x_train[train_index], x_train[test_index]
                y_cv_train, y_cv_test = y_train[train_index], y_train[test_index]
                self.n_clusters = value 
                self.fit(x_cv_train, y_cv_train) 
                y_pred = self.predict(x_cv_test) 
                score = accuracy_score(y_cv_test, y_pred, normalize=True)
                if (best_score < score):
                    best_score = score
                    best_parameter = value 
        
        return {"Best Score":best_score, "Best Parameter":best_parameter}
        
    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"clusters": self.n_clusters}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self