# Inspired by: "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow"

# Uses supervised methods. 
class Model_Evaluater: 

    def __init__(self, x_train, y_train, x_test, y_test, model, colors):
        from sklearn.decomposition import PCA 
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = model
        self.colors = colors 
        pca = PCA(n_components=2) 
        self.x_train_pca = pca.fit_transform(self.x_train) 
        self.x_test_pca = pca.fit_transform(self.x_test) 

    def visualize_confusion_matrix(self):  
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt 
        import numpy as np 
        
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
                        
    def print_classification_report(self):
        from sklearn import metrics 
        
        self.model.fit(self.x_train, self.y_train)
        y_test_pred = self.model.predict(self.x_test) 

        print(
            f"Classification report for classifier {self.model}:\n"
            f"{metrics.classification_report(self.y_test, y_test_pred)}\n"
        )

    def display_precision_recall_curve(self):
        from sklearn.metrics import precision_recall_curve
        import matplotlib.pyplot as plt 
        
        self.model.fit(self.x_train, self.y_train)
        y_test_pred = self.model.predict(self.x_test) 
        
        precisions, recalls, thresholds = precision_recall_curve(self.y_test, y_test_pred)
        plt.plot(thresholds, precisions[:, -1], "b--", label="Precision")
        plt.plot(thresholds, recalls[:, -1], "g-", label="Recall")
        plt.show() 
        
    # Must be 2-Dimensional, i.e. using PCA.    
    def display_cluster_allocations(self):
        import numpy as np 
        import matplotlib
        import matplotlib.pyplot as plt 
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
        plt.title("Centroids versus actual classification")
        plt.axis("tight")
        
        plt.show() 
        
        # Plot the actual class versus class centroids. 
        plt.scatter(self.x_test_pca[:, 0], self.x_test_pca[:, 1], c=y_test_pred, cmap=matplotlib.colors.ListedColormap(self.colors), edgecolor="k", s=20)
        plt.title("Predicted Classification")
        plt.axis("tight")
        
        plt.show() 

    # Must be 2-Dimensional, i.e. using PCA.       
    def display_scatter(self):
        import matplotlib
        import matplotlib.pyplot as plt 
        import numpy as np 
        labels = np.arange(10)
        plt.scatter(self.x_train_pca[:, 0],  self.x_train_pca[:, 1], c=self.y_train, cmap=matplotlib.colors.ListedColormap(self.colors))

        cb = plt.colorbar()
        loc = np.arange(0,max(self.y_train),max(self.y_train)/float(len(labels)))
        cb.set_ticks(loc)
        cb.set_ticklabels(labels)
        
    def describe_model(self): 
        self.model.fit(self.x_train, self.y_train)
        
        print("Params: ", self.model.get_params())
        print("Mean accuracy on test data {:.2f}%".format(self.model.score(self.x_test, self.y_test) * 100))
