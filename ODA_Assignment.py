# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 12:40:32 2021

@author: Trill
"""

# %% Setup data. 
from scipy.io import loadmat 
from mnist import MNIST 
import numpy as np 
import os 
os.chdir('C:/Users/Trill/Desktop/9.Semester/ODA/ODAProject')
print(os.getcwd())


data = MNIST("data")
x_train, y_train = np.array(data.load_training())
x_test, y_test = np.array(data.load_testing())

# 60k images, each image has 784 featues (28 x 28). 
print("Images shape: ", x_train.shape)


#data = loadmat('mnist_loaded.mat')

#x_train = data["train_images"].transpose()
#y_train = data["train_labels"].ravel()

#x_test = data["test_images"].transpose() 
#y_test = data["test_labels"].ravel()


# %% Setup 2D Data by applying PCA. 
from sklearn.decomposition import PCA 

pca = PCA(n_components=2) 
x_train_pca = pca.fit_transform(x_train) 
x_test_pca = pca.fit_transform(x_test) 
print("Proportion of variance: ", pca.explained_variance_)
print("Singular Decomposition Values: ", pca.singular_values_) 
print("Images with 2 PCA components (shape)", x_train_pca.shape)

# %% Visualize Data 
import matplotlib
import matplotlib.pyplot as plt 
# Images and corresponding 
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i].reshape((28,28)), cmap=plt.cm.binary)
    plt.xlabel(y_train[i])
plt.show()

# Images with PCA reduction (2 features) 
# Inspired by: https://stackoverflow.com/questions/12487060/matplotlib-color-according-to-class-labels
colors = ['red', 'green', 'blue', 'purple', 'yellow', 'pink', 'cyan', 'teal', 'violet']
labels = np.arange(10)
plt.scatter(x_train_pca[:, 0],  x_train_pca[:, 1], c=y_train, cmap=matplotlib.colors.ListedColormap(colors))

cb = plt.colorbar()
loc = np.arange(0,max(y_train),max(y_train)/float(len(labels)))
cb.set_ticks(loc)
cb.set_ticklabels(labels)

# %% Nearest class centroid classifier 
# Using scikit-learns implementation: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html
# Visualization inspired by: https://scikit-learn.org/stable/auto_examples/neighbors/plot_nearest_centroid.html#sphx-glr-auto-examples-neighbors-plot-nearest-centroid-py
# Each class is represented by its centroid
# New samples are classified based on nearest class centroid. 
# Data can only be vizualised in 2D. 

from sklearn.neighbors import NearestCentroid
h = 0.2 # Step size

X = x_train_pca 
y = y_train

n_c_model = NearestCentroid() 
n_c_model.fit(X, y) 

y_pred = n_c_model.predict(X)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = n_c_model.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=matplotlib.colors.ListedColormap(colors))

# Plot the actual class versus class centroids. 
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(colors), edgecolor="k", s=20)
plt.title("Centroids versus actual classification")
plt.axis("tight")

plt.show() 

# Plot the actual class versus class centroids. 
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=matplotlib.colors.ListedColormap(colors), edgecolor="k", s=20)
plt.title("Predicted Classification")
plt.axis("tight")

plt.show() 

# %% Nearest Class centroid classifier WITHOUT pca. 

n_c_model = NearestCentroid() 
n_c_model.fit(x_train, y_train) 

print("Params: ", n_c_model.get_params())
print("Mean accuracy on test data {}%".format(n_c_model.score(x_test, y_test) * 100))

# Scoring: https://scikit-learn.org/stable/modules/model_evaluation.html

# %% Evaluate Model
# Inspired by: "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow"

# Uses supervised methods. 
class ModelEvaluater: 
    
    def visualize_confusion_matrix(self, x_train, y_train, model, k_fold):  
        from sklearn.model_selection import cross_val_predict 
        from sklearn.metrics import confusion_matrix
        import numpy as np 
        
        y_train_pred = cross_val_predict(model, x_train, y_train, cv=k_fold)
        
        conf_mx = confusion_matrix(y_train, y_train_pred)
        plt.matshow(conf_mx, cmap=plt.cm.gray)
        plt.show() 
        
        row_sums = conf_mx.sum(axis=1, keepdims=True)
        norm_conf_mx = conf_mx / row_sums 
        np.fill_diagonal(norm_conf_mx, 0)
        plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
        plt.show() 
        
    def print_classification_report(self, x_train, y_train, model, k_fold):
        from sklearn import metrics 
        from sklearn.model_selection import cross_val_predict
        
        y_train_pred = cross_val_predict(model, x_train, y_train, cv=k_fold)

        print(
            f"Classification report for classifier {model}:\n"
            f"{metrics.classification_report(y_train, y_train_pred)}\n"
        )
        
    def display_precision_recall_curve(self, x_train, y_train, model, k_fold):
        from sklearn.model_selection import cross_val_predict 
        from sklearn.metrics import precision_recall_curve
        
        y_train_pred = cross_val_predict(model, x_train, y_train, cv=k_fold)
        precisions, recalls, thresholds = precision_recall_curve(y_train, y_train_pred)
        plt.plot(thresholds, precisions[:, -1], "b--", label="Precision")
        plt.plot(thresholds, recalls[:, -1], "g-", label="Recall")
        plt.show() 
        
    # Must be 2-Dimensional, i.e. using PCA.    
    def display_cluster_allocations(self, x_train, y_train, model, k_fold, colors):
        import numpy as np 
        from sklearn.model_selection import cross_val_predict
        h = 0.2 # Step size

        X = x_train 
        y = y_train
        
        n_c_model = NearestCentroid() 
        n_c_model.fit(X, y) 
        
        y_pred = n_c_model.predict(X)
        
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = n_c_model.predict(np.c_[xx.ravel(), yy.ravel()])
        
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=matplotlib.colors.ListedColormap(colors))
        
        # Plot the actual class versus class centroids. 
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(colors), edgecolor="k", s=20)
        plt.title("Centroids versus actual classification")
        plt.axis("tight")
        
        plt.show() 
        
        # Plot the actual class versus class centroids. 
        plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=matplotlib.colors.ListedColormap(colors), edgecolor="k", s=20)
        plt.title("Predicted Classification")
        plt.axis("tight")
        
        plt.show() 

        
    def display_scatter(self, x_train, y_train, colors):
        # Images with PCA reduction (2 features) 
        # Inspired by: https://stackoverflow.com/questions/12487060/matplotlib-color-according-to-class-labels
        labels = np.arange(10)
        plt.scatter(x_train_pca[:, 0],  x_train_pca[:, 1], c=y_train, cmap=matplotlib.colors.ListedColormap(colors))

        cb = plt.colorbar()
        loc = np.arange(0,max(y_train),max(y_train)/float(len(labels)))
        cb.set_ticks(loc)
        cb.set_ticklabels(labels)
    
colors = ['red', 'green', 'blue', 'purple', 'yellow', 'pink', 'cyan', 'teal', 'violet']
m_eval = ModelEvaluater()
m_eval.visualize_confusion_matrix(x_train, y_train, n_c_model, 5)
m_eval.print_classification_report(x_train, y_train, n_c_model, 5)
m_eval.display_cluster_allocations(x_train_pca, y_train, n_c_model, 5, colors)
m_eval.display_scatter(x_train_pca, y_train, colors)


