from functions_1119 import *
from scipy.optimize import minimize
from scipy.integrate import quad
 # Simpson's rule for numerical integration
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
from datetime import datetime

import warnings
warnings.filterwarnings('ignore', 'Values in x were outside bounds during a minimize step, clipping to bounds')


np.random.seed(42)


df = pd.read_stata("data_cont_training_YF_jan1824.dta")
col_name= ['edu', 'prevearn', 'earnings','wkforpy', 'weekswrk','adultm', 'age', 'bfeduca', 'bfvococc',
       'bfabeesl', 'bfwforpy', 'bfjbclub', 'bfmarsta', 'bftotinc', 'bfnoafdc',
       'bfpbhous', 'bfmandat', 'portuges', 'tagalog', 'nohs', 'numdeg', 
       'yearearn', 'recafdc', 'anyafdcy', 'corpus', 'cedar', 'coosa', 'heart', 'fortw', 'jersey',
       'jackson', 'larimar', 'decatur', 'nwminn', 'montana', 'omaha', 'marion',
       'oakland', 'provid', 'springf', 'ctonly', 'ojtonly', 'actothr', 'adultf', 'youthm', 'youthn', 'numrecs',
       'numothr', 'white', 'black', 'hispanic', 'native', 'asian', 'wht_blk',
       'w_b_his', 'female', 'ramonth', 'cohort',  'phonehom', 'hour_train']
df = df[col_name]

for x in df.columns:
    if np.max(df[x]) == 9 and len(np.unique(df[x])) == 3:
       df.loc[df[x]==9, x] = np.nan 
    elif np.max(df[x]) == 99:
        df.loc[df[x]==99, x] = np.nan 
    elif np.max(df[x]) == 999:
        df.loc[df[x]==999, x]= np.nan 
    elif np.max(df[x]) == 9999:
        df.loc[df[x]==9999, x] = np.nan 


keep_column = df.columns[df.isnull().sum() == 0]
data = df[keep_column]

# restrict the sample to be plausible
data = data[data['prevearn']<=20000]
data = data[data['hour_train']<1360]
data = data[data["earnings"]<=80000]
data["week_train"] = data["hour_train"]/40


data_main, data_pilot = train_test_split(data, test_size=0.02, random_state=42)
y_pilot = np.array(data_pilot["earnings"])
T_pilot = np.array(data_pilot["hour_train"])
X_pilot = np.array(data_pilot.drop(["earnings", "hour_train", "week_train"], axis = 1))

lb = 1
ub = np.log(200)
bnd = (lb, ub)
kernel_type = "sinc"
use_kernel = 0
dr = 0

pilot_dict = {"y": y_pilot, "T": T_pilot}

A_hat, r_hat = est_A_r(pilot_dict, bnd, kernel_type , use_kernel, dr)

y = np.array(data["earnings"])
print(y.shape)
T = np.array(data["week_train"])
X = np.array(data.drop(["earnings", "hour_train", "week_train"], axis = 1))
X_policy = np.array(data[['edu', 'prevearn', 'weekswrk']])
T_mean = np.mean(T)

t_lo = np.min(data_main["week_train"])
t_hi = np.max(data_main["week_train"])
# X_policy = np.c_[X_policy, np.ones(len(y))]


input_data = {'y': y,  'x':  X, 'x_pol': X_policy, 'T': T,  'cost': 100, 'A_hat': A_hat, 'r_hat': r_hat, 't_lo': t_lo, 't_hi': t_hi}

# Save input dictionary to pickle file
date_time = datetime.now().strftime("%m%d")
with open(f'input_rad_{date_time}.pkl', 'wb') as f:
    pickle.dump(input_data, f)

