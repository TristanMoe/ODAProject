# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 08:19:52 2021

@author: Trill
"""
from sklearn.model_selection import GridSearchCV
from mnist import MNIST 
from scipy.io import loadmat 
import numpy as np 
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
        
class Utils:
    def __init__(self, use_mnist):
        if (use_mnist):
            data = MNIST("data_mnist")
            self.x_train, self.y_train = np.array(data.load_training())
            self.x_test, self.y_test = np.array(data.load_testing())

            self.x_train = np.array([i for i in self.x_train]).reshape(-1, 28*28) / 255
            self.x_test = np.array([i for i in self.x_test]).reshape(-1, 28*28) / 255 
            self.y_train = np.array([i for i in self.y_train], dtype=np.uint8).ravel()
            self.y_test = np.array([i for i in self.y_test], dtype=np.uint8).ravel() 
            self.name = "MNIST"
            self.colors = ['red', 'green', 'blue', 'purple', 'yellow', 'pink', 'cyan', 'teal', 'violet']
        else:
            data = loadmat('data_orl/orl_data.mat')["data"].transpose()
            labels = loadmat('data_orl/orl_lbls.mat')["lbls"].ravel()
            lbl = np.unique(labels)
            self.name = "ORL"

            self.colors = [None] * len(lbl)
            for idx in range(len(lbl)):
                self.colors[idx] = np.random_color=list(np.random.choice(range(255),size=3))

            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data, labels, test_size=0.33, random_state=0, stratify=labels)


            data_train = [False] * len(lbl)
            data_test = [False] * len(lbl)
            for idx, l in enumerate(lbl):
                data_train[idx] = l in self.y_train
                data_test[idx] = l in self.y_test
            
            if (all(data_train) and all(data_test)):
                print("All classes are present")
            else:
                print("NOT all classes are present")
                
    def get_name(self):
        return self.name
    
    def fetch_data(self):
        return self.x_train, self.y_train, self.x_test, self.y_test
    
    def get_colors(self):
        return self.colors 
    
    # Setup 2D Data by applying PCA. 
    def fetch_pca(self):
        print("Images shape: ", self.x_train.shape)

        pca = PCA(n_components=2) 
        self.x_train_pca = pca.fit_transform(self.x_train) 
        self.x_test_pca = pca.fit_transform(self.x_test) 
        print("Proportion of variance: ", pca.explained_variance_)
        print("Singular Decomposition Values: ", pca.singular_values_) 
        print("Images with 2 PCA components (shape)", self.x_train_pca.shape)
        return self.x_train_pca, self.x_test_pca
        
    def visualize_data(self):   
        # Images and corresponding 
        plt.figure(figsize=(10,10))
        for i in range(25):
            plt.subplot(5,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.x_train[i].reshape((28,28)), cmap=plt.cm.binary)
            plt.xlabel(self.y_train[i])
        plt.show()
            
    def grid_search(self, model, x_train, y_train, parameters):         
        grid_model = GridSearchCV(model, parameters)
        grid_model.fit(x_train, y_train)
        return grid_model.cv_results_
