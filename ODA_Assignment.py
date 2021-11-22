# %% Setup data. 
import os 
import numpy as np 
os.chdir('C:/Users/Trill/Desktop/9.Semester/ODA/ODAProject')
print(os.getcwd())

import Utils
pca = False 
util = Utils.Utils(False) 
dataname = util.get_name() 
colors = util.get_colors() 

x_train, y_train, x_test, y_test = util.fetch_data()
x_train_pca, x_test_pca = util.fetch_pca() 

#util.visualize_data()

print() 
print() 

# Shorten training time 
#x_train = x_train[0:1000]
#y_train = y_train[0:1000]

# %% Nearest Centroid Classifier 
import Model_Evaluater as m_e
from sklearn.neighbors import NearestCentroid

n_c_model = NearestCentroid() 
m_eval = m_e.Model_Evaluater(x_train, y_train, x_test, y_test, n_c_model, colors, "Nearest Centroid")

#m_eval.display_cluster_allocations()
n_c_score = m_eval.print_classification_report(pca)
#m_eval.visualize_confusion_matrix()
# No grid search 

# %% Subclass Nearest Centroid Classifier
import Nearest_Centroid as n_c 
import Model_Evaluater as m_e

parameters = {2, 3}#, 4, 5}
result = n_c.Subclass_Nearest_Centroid().grid_search_kfold(x_train, y_train, parameters)
print("Best hyper paramenter subclusters:", result["Best Parameter"])

s_n_c_model = n_c.Subclass_Nearest_Centroid(result["Best Parameter"])
m_eval = m_e.Model_Evaluater(x_train, y_train, x_test, y_test, s_n_c_model, colors, "Subclass Centroid")
        
#m_eval.display_cluster_allocations()
s_n_c_score = m_eval.print_classification_report(pca)
s_n_c_score = m_eval.describe_model()
#m_eval.visualize_confusion_matrix()

# %% Nearest Neighbor Classifier
from sklearn.neighbors import KNeighborsClassifier
import Model_Evaluater as m_e
import numpy as np 

parameters = {"n_neighbors":(1, 2, 3, 4, 5, 10)}
result = util.grid_search(KNeighborsClassifier(), x_train, y_train, parameters)

best_index = np.where(result["mean_test_score"] == np.amax(result["mean_test_score"]))
best_parameter = result["param_n_neighbors"][best_index][0]
print("Best hyper paramenter neighbors:", best_parameter)

k_n_model = KNeighborsClassifier(n_neighbors=best_parameter)
m_eval = m_e.Model_Evaluater(x_train, y_train, x_test, y_test, k_n_model, colors, "K-Nearest Neighbor")
        
#m_eval.display_cluster_allocations()
k_n_score = m_eval.print_classification_report(pca)
#m_eval.visualize_confusion_matrix()

# %% Perceptron using Backpropagation Classifier 
# https://scikit-learn.org/stable/modules/neural_networks_supervised.html
from sklearn.linear_model import SGDClassifier
import Model_Evaluater as m_e

# TODO: Hyper parameter learning rate
parameters = {"eta0":(1, 0.1, 0.01, 0.001)}
result = util.grid_search(SGDClassifier(loss='hinge', learning_rate="constant", penalty=None, alpha=0), x_train, y_train, parameters)

best_index = np.where(result["mean_test_score"] == np.amax(result["mean_test_score"]))
best_parameter = result["param_eta0"][best_index][0]
print("Best hyper paramenter eta0 (HINGE):", best_parameter)

p_back_model = SGDClassifier(loss='hinge', eta0=best_parameter, learning_rate="constant", penalty=None, alpha=0)
m_eval = m_e.Model_Evaluater(x_train, y_train, x_test, y_test, p_back_model, colors, "Perceptron w. Backpropagation")
        
#m_eval.display_cluster_allocations()
p_back_score = m_eval.print_classification_report(pca)
#m_eval.visualize_confusion_matrix()

# %% Perceptron using MSE classifier 
from sklearn.linear_model import SGDClassifier
import Model_Evaluater as m_e

# See parameters for optimization (i.e. GridSearch)
# TODO: Hyper parameter learning rate (change eta0)
parameters = {"eta0":(1, 0.1, 0.01, 0.001, 0.0001)}
result = util.grid_search(SGDClassifier(loss='squared_error', learning_rate="constant", penalty=None, alpha=0), x_train, y_train, parameters)

best_index = np.where(result["mean_test_score"] == np.amax(result["mean_test_score"]))
best_parameter = result["param_eta0"][best_index][0]
print("Best hyper paramenter eta0 (MSE):", best_parameter)

p_mse_model = SGDClassifier(loss='squared_error', eta0=best_parameter, learning_rate="constant", penalty=None, alpha=0)
m_eval = m_e.Model_Evaluater(x_train, y_train, x_test, y_test, p_mse_model, colors, "Perceptron w. MSE")
        
#m_eval.display_cluster_allocations()
p_mse_score = m_eval.print_classification_report(pca)
#m_eval.visualize_confusion_matrix()

# %% Visualize data
import matplotlib.pyplot as plt 
# Scores
x_axis = [n_c_score, s_n_c_score, k_n_score, p_back_score, p_mse_score]
y_axis = ["NN", "SNC", "KNN", "PB", "PMSE"]
x_ticks = np.arange(len(y_axis))

plt.bar(x_ticks, height=x_axis, color=['black', 'red', 'green', 'blue', 'cyan'])
plt.xticks(x_ticks, y_axis)
plt.title("Model Scores for {}".format(dataname))
plt.show()

# Hyper parameters
x_axis = result["mean_test_score"]
y_axis = result["param_eta0"]
x_ticks = np.arange(len(y_axis))

plt.bar(x_ticks, height=x_axis, color=['black', 'red', 'blue', 'green'])
plt.xticks(x_ticks, y_axis)
plt.title("Hyper paramenter scores for perceptron using MSE")
plt.show()
