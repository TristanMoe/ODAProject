# Inspired by: "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow"

import matplotlib
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
import numpy as np 
from sklearn import metrics 
from sklearn.metrics import accuracy_score
# Uses supervised methods. 
class Model_Evaluater: 

    def __init__(self, x_train, y_train, x_test, y_test, model, colors, name):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = model
        self.colors = colors 
        pca = PCA(n_components=2) 
        self.x_train_pca = pca.fit_transform(self.x_train) 
        self.x_test_pca = pca.fit_transform(self.x_test) 
        self.name = name

    def visualize_confusion_matrix(self):
        self.model.fit(self.x_train, self.y_train)
        y_test_pred = self.model.predict(self.x_test) 
        
        conf_mx = confusion_matrix(self.y_test, y_test_pred)
        plt.matshow(conf_mx, cmap=plt.cm.gray)
        plt.show() 
        
        row_sums = conf_mx.sum(axis=1, keepdims=True)
        norm_conf_mx = conf_mx / row_sums 
        np.fill_diagonal(norm_conf_mx, 0)
        plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
        plt.show() 
                        
    def print_classification_report(self, pca = False):
        if (pca):
            self.model.fit(self.x_train_pca, self.y_train)
            y_test_pred = self.model.predict(self.x_test_pca) 
        else:
            self.model.fit(self.x_train, self.y_train)
            y_test_pred = self.model.predict(self.x_test) 

        classification_report = metrics.classification_report(self.y_test, y_test_pred)
        print(
            f"Classification report for classifier {self.model}:\n"
            f"{classification_report}\n"
        )
        
        return accuracy_score(self.y_test, y_test_pred, normalize=True)

    def display_precision_recall_curve(self):        
        self.model.fit(self.x_train, self.y_train)
        y_test_pred = self.model.predict(self.x_test) 
        
        precisions, recalls, thresholds = precision_recall_curve(self.y_test, y_test_pred)
        plt.plot(thresholds, precisions[:, -1], "b--", label="Precision")
        plt.plot(thresholds, recalls[:, -1], "g-", label="Recall")
        plt.show() 
    
    # Must be 2-Dimensional, i.e. using PCA.    
    def display_cluster_allocations(self):
        h = 0.2 # Step size
        
        self.model.fit(self.x_train_pca, self.y_train)
        y_test_pred = self.model.predict(self.x_test_pca) 
        
        x_min, x_max = self.x_test_pca[:, 0].min() - 1, self.x_test_pca[:, 0].max() + 1
        y_min, y_max = self.x_test_pca[:, 1].min() - 1, self.x_test_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=matplotlib.colors.ListedColormap(self.colors))
        
        # Plot the actual class versus class centroids. 
        plt.scatter(self.x_test_pca[:, 0], self.x_test_pca[:, 1], c=self.y_test, cmap=matplotlib.colors.ListedColormap(self.colors), edgecolor="k", s=20)
        plt.title("TRAIN centroids versus actual TEST classification \n{}".format(self.name))
        plt.axis("tight")
        
        plt.show() 
        
        # Plot the actual class versus class centroids. 
        plt.scatter(self.x_test_pca[:, 0], self.x_test_pca[:, 1], c=y_test_pred, cmap=matplotlib.colors.ListedColormap(self.colors), edgecolor="k", s=20)
        plt.title("Predicted Classification \n{}".format(self.name))
        plt.axis("tight")
        
        plt.show() 
    
    # Taken from: https://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_iris.html#sphx-glr-auto-examples-linear-model-plot-sgd-iris-py
    def display_decision_surface(self):
        h = 0.02  # step size in the mesh

        X = self.x_train_pca
        y = self.y_train
        classes = np.unique(y)
        self.model.fit(X, y)
    
        # create a mesh to plot in
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.axis("tight")

        # Plot also the training points
        for i, color in zip(classes, self.colors):
            idx = np.where(y == i)
            plt.scatter(
                X[idx, 0],
                X[idx, 1],
                c=color,
                label=classes[i],
                cmap=plt.cm.Paired,
                edgecolor="black",  
                s=20,
                )
        
        plt.title("Decision surface of multi-class {}".format(self.name))
        plt.axis("tight")

        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()
        coef = self.model.coef_
        intercept = self.model.intercept_
        print(coef)
        print(intercept)
    

        def plot_hyperplane(c, color):
            def line(x0):
                return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]
            
            plt.plot([xmin, xmax], [line(xmin), line(xmax)], ls="--", color=color)


        for c, color in zip(classes, self.colors):
            plot_hyperplane(c, color)
        
        plt.legend()
        plt.show()

    # Must be 2-Dimensional, i.e. using PCA.       
    def display_scatter(self): 
        labels = np.arange(10)
        plt.title("Train data and class label")
        plt.scatter(self.x_train_pca[:, 0],  self.x_train_pca[:, 1], c=self.y_train, cmap=matplotlib.colors.ListedColormap(self.colors))

        cb = plt.colorbar()
        loc = np.arange(0,max(self.y_train),max(self.y_train)/float(len(labels)))
        cb.set_ticks(loc)
        cb.set_ticklabels(labels)
        
    def describe_model(self): 
        self.model.fit(self.x_train, self.y_train)
        y_pred = self.model.predict(self.x_test)
        score = accuracy_score(self.y_test, y_pred, normalize=True)
        print("Params: ", self.model.get_params())
        print("Mean accuracy on test data {:.2f}%".format(score))
        return score 