# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 08:19:52 2021

@author: Trill
"""

class Utils:
    def __init__(self, use_mnist):
        from mnist import MNIST 
        from scipy.io import loadmat 
        
        if (use_mnist):
            data = MNIST("data")
            self.x_train, self.y_train = np.array(data.load_training())
            self.x_test, self.y_test = np.array(data.load_testing())
        else:
            data = loadmat('mnist_loaded.mat')
            self.x_train = data["train_images"].transpose()
            self.y_train = data["train_labels"].ravel()

            self.x_test = data["test_images"].transpose() 
            self.y_test = data["test_labels"].ravel()
    
    def fetch_data(self):
        return self.x_train, self.y_train, self.x_test, self.y_test
    
    # %% Setup 2D Data by applying PCA. 
    def fetch_pca(self):
        from sklearn.decomposition import PCA 
        print("Images shape: ", self.x_train.shape)

        pca = PCA(n_components=2) 
        self.x_train_pca = pca.fit_transform(self.x_train) 
        self.x_test_pca = pca.fit_transform(self.x_test) 
        print("Proportion of variance: ", pca.explained_variance_)
        print("Singular Decomposition Values: ", pca.singular_values_) 
        print("Images with 2 PCA components (shape)", self.x_train_pca.shape)
        return self.x_train_pca, self.x_test_pca
        
    def visualize_data(self):
        import matplotlib
        import matplotlib.pyplot as plt 
        
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
