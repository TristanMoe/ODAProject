# %% Setup data. 
import os 
os.chdir('C:/Users/Trill/Desktop/9.Semester/ODA/ODAProject')
print(os.getcwd())

import Utils
util = Utils.Utils(True) 

x_train, y_train, x_test, y_test = util.fetch_data()
x_train_pca, x_test_pca = util.fetch_pca() 
colors = ['red', 'green', 'blue', 'purple', 'yellow', 'pink', 'cyan', 'teal', 'violet']

util.visualize_data()

# %% Nearest Centroid Classifier 
import Model_Evaluater as m_e
from sklearn.neighbors import NearestCentroid

n_c_model = NearestCentroid() 
m_eval = m_e.Model_Evaluater(x_train, y_train, x_test, y_test, n_c_model, colors)

#m_eval.display_cluster_allocations()
n_c_report = m_eval.print_classification_report()
#m_eval.visualize_confusion_matrix()

# %% Subclass Nearest Centroid Classifier
import Nearest_Centroid as n_c 
import Model_Evaluater as m_e

clusters = 2
s_n_c_model = n_c.Subclass_Nearest_Centroid(clusters)
m_eval = m_e.Model_Evaluater(x_train, y_train, x_test, y_test, s_n_c_model, colors)
        
#m_eval.display_cluster_allocations()
s_n_c_report = m_eval.print_classification_report()
#m_eval.visualize_confusion_matrix()

# %% Nearest Neighbor Classifier
from sklearn.neighbors import KNeighborsClassifier
import Model_Evaluater as m_e

neighbors = 10 # Different values should be tested. 
k_n_model = KNeighborsClassifier(n_neighbors=neighbors)
m_eval = m_e.Model_Evaluater(x_train, y_train, x_test, y_test, k_n_model, colors)
        
#m_eval.display_cluster_allocations()
k_n_report = m_eval.print_classification_report()
#m_eval.visualize_confusion_matrix()

# %% Perceptron using Backpropagation Classifier 
# https://scikit-learn.org/stable/modules/neural_networks_supervised.html
from sklearn.linear_model import SGDClassifier
import Model_Evaluater as m_e

# TODO: Hyper parameter learning rate
p_back_model = SGDClassifier(loss='hinge', eta0=1, learning_rate="constant", penalty=None)
m_eval = m_e.Model_Evaluater(x_train, y_train, x_test, y_test, p_back_model, colors)
        
#m_eval.display_cluster_allocations()
p_back_report = m_eval.print_classification_report()
#m_eval.visualize_confusion_matrix()


# %% Perceptron using MSE classifier 
from sklearn.linear_model import SGDClassifier
import Model_Evaluater as m_e

# See parameters for optimization (i.e. GridSearch)
# TODO: Hyper parameter learning rate
p_mse_model = SGDClassifier(loss='squared_error', eta0=1, learning_rate="constant", penalty=None)
m_eval = m_e.Model_Evaluater(x_train, y_train, x_test, y_test, p_mse_model, colors)
        
#m_eval.display_cluster_allocations()
p_mse_report = m_eval.print_classification_report()
#m_eval.visualize_confusion_matrix()