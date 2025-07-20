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
with open(f'input_rad_{date_time}.pkl', 'rb') as f:
    input_data = pickle.load(f)



num_fold = 2
method = 'rf'

f_score_est, model_y_dict, cv_idx_dic = est_model(input_data, num_fold = 2)
input_data['f_score_est'] = f_score_est
input_data['model_y_dict'] = model_y_dict
input_data['cv_idx_dic'] = cv_idx_dic





def ind_task(k, h, input_data):
    
    params = (k, h)
    obj, penal_obj, beta = evaluation_function_rad(params, input_data)

    return (obj, penal_obj, beta)

def task_function(task):
    k, k2 = task
    h = k2/10
    k_rad_dic = {}
    result = ind_task(k, h, input_data)
    k_rad_dic["welfare"] = result[0]
    k_rad_dic["penalized_obj"] = result[1]
    k_rad_dic["coefficient"] = result[2]
        
    file_name = f'results_rad_241120/results_{k}_{k2}.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(k_rad_dic, f)


cpu_num = 8
if __name__ == "__main__":
    k_list = range(2, 3)
    k2_list = range(9, 10)
    tasks = list(itertools.product(k_list, k2_list))
    with Pool(cpu_num) as pool:
        pool.map(task_function, tasks)
