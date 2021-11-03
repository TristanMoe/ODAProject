# %% Setup data. 
import os 
os.chdir('C:/Users/Trill/Desktop/9.Semester/ODA/ODAProject')
print(os.getcwd())

import Utils
util = Utils.Utils(False) 

x_train, y_train, x_test, y_test = util.fetch_data()
x_train_pca, x_test_pca = util.fetch_pca() 

#util.visualize_data()

# %% Model Evaluation. 
import Model_Evaluater as m_e
from sklearn.neighbors import NearestCentroid

n_c_model = NearestCentroid() 

colors = ['red', 'green', 'blue', 'purple', 'yellow', 'pink', 'cyan', 'teal', 'violet']
m_eval = m_e.Model_Evaluater(x_train_pca, y_train, x_test_pca, y_test, n_c_model, colors)
#m_eval.visualize_confusion_matrix()
#m_eval.print_classification_report()
m_eval.display_cluster_allocations()
#m_eval.display_scatter()
#m_eval.describe_model()

# %% Subclass Nearest Centroid 
import Nearest_Centroid as nc 

s_n_c_model = nc.Subclass_Nearest_Centroid(2)
s_n_c_model.fit(x_train, y_train) 
y_pred = s_n_c_model.predict(x_test)
m_eval = m_e.Model_Evaluater(x_train, y_train, x_test, y_test, s_n_c_model, colors)
        
# %% 
#m_eval.display_scatter() 
m_eval.display_cluster_allocations()