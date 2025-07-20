from functions_1119 import *
from scipy.optimize import minimize
from scipy.integrate import quad
 # Simpson's rule for numerical integration
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
from datetime import datetime
from multiprocess import Pool 
import multiprocessing
import itertools

import warnings

warnings.filterwarnings("ignore", message="resource_tracker*")
warnings.filterwarnings('ignore', 'Values in x were outside bounds during a minimize step, clipping to bounds')

# Set random seed for reproducibility
np.random.seed(42)

# Load input dictionary from pickle file
date_time = datetime.now().strftime("%m%d")
with open(f'input_1123.pkl', 'rb') as f:
    input = pickle.load(f)

data_est_dict = input['data_estimation']
data_test_dict = input['data_testing']


num_fold = 2
method = 'rf'

f_score_est, model_y_dict, cv_idx_dic_train = est_model(data_est_dict, num_fold = 2)
data_est_dict['f_score_est'] = f_score_est
data_est_dict['model_y_dict'] = model_y_dict
data_est_dict['cv_idx_dic'] = cv_idx_dic_train

f_score_test, model_y_dict, cv_idx_dic_test = est_model(data_test_dict, num_fold = 2)
data_test_dict['f_score_est'] = f_score_test
data_test_dict['model_y_dict'] = model_y_dict
data_test_dict['cv_idx_dic'] = cv_idx_dic_test



def ind_task(k, h, data_est_dict, data_test_dict):
    
    params = (k, h)

    results = optimization_function(params, data_est_dict)
    beta = results['beta']
    obj_test, penal_obj = evaluation_function(results, params, data_test_dict)

    return (obj_test, penal_obj, beta)

def task_function(task):
    k, k2 = task
    h = k2/20
    k_rad_dic = {}
    result = ind_task(k, h, data_est_dict, data_test_dict)
    k_rad_dic["welfare_test"] = result[0]
    k_rad_dic["penalized_obj"] = result[1]
    k_rad_dic["coefficient"] = result[2]
        
    # date_time = datetime.now().strftime("%m%d")
    file_name = f'results_holdout_241123/results_{k}_{k2}.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(k_rad_dic, f)


cpu_num = 8
if __name__ == "__main__":
    k_list = range(1, 11)
    k2_list = range(1, 20)
    tasks = list(itertools.product(k_list, k2_list))
    with Pool(cpu_num) as pool:
        pool.map(task_function, tasks)
