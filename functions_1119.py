import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import scipy.stats as stats
from scipy import interpolate
from scipy.optimize import minimize  
from scipy.optimize import LinearConstraint                                                                                                                                                                                                                                                                   
from scipy.spatial.distance import cdist
from scipy.misc import derivative
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
# from gurobipy import Model, GRB
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import gaussian_kde
from scipy.integrate import simpson 
import gurobipy as gp
import gurobi_ml
import gurobipy_pandas as gppd
from joblib import Parallel, delayed
import  collections
import statsmodels.api as sm
import os
import time
import sys
import math
import datetime
import pickle
import csv
from tqdm import tqdm
# import rfcde
# Define kernel functions


class ConditionalNearestNeighborsKDE(BaseEstimator):
    """Conditional Kernel Density Estimation using nearest neighbors.

    This class implements a Conditional Kernel Density Estimation by applying
    the Kernel Density Estimation algorithm after a nearest neighbors search.

    It allows the use of user-specified nearest neighbor and kernel density
    estimators or, if not provided, defaults will be used.

    Parameters
    ----------
    nn_estimator : NearestNeighbors instance, default=None
        A pre-configured instance of a `~sklearn.neighbors.NearestNeighbors` class
        to use for finding nearest neighbors. If not specified, a
        `~sklearn.neighbors.NearestNeighbors` instance with `n_neighbors=100`
        will be used.

    kde_estimator : KernelDensity instance, default=None
        A pre-configured instance of a `~sklearn.neighbors.KernelDensity` class
        to use for estimating the kernel density. If not specified, a
        `~sklearn.neighbors.KernelDensity` instance with `bandwidth="scott"`
        will be used.
    """

    def __init__(self, nn_estimator=None, kde_estimator=None):
        self.nn_estimator = nn_estimator
        self.kde_estimator = kde_estimator

    def fit(self, X, y=None):
        if self.nn_estimator is None:
            self.nn_estimator_ = NearestNeighbors(n_neighbors=5)
        else:
            self.nn_estimator_ = clone(self.nn_estimator)
        self.nn_estimator_.fit(X, y)
        self.y_train_ = y
        return self

    def predict(self, X):
        """Predict the conditional density estimation of new samples.

        The predicted density of the target for each sample in X is returned.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be estimated, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        kernel_density_list : list of len n_samples of KernelDensity instances
            Estimated conditional density estimations in the form of
            `~sklearn.neighbors.KernelDensity` instances.
        """
        _, ind_X = self.nn_estimator_.kneighbors(X)
        if self.kde_estimator is None:
            kernel_density_list = [
                KernelDensity(bandwidth='scott').fit(self.y_train_[ind].reshape(-1, 1))
                for ind in ind_X
            ]
        else:
            kernel_density_list = [
                clone(self.kde_estimator).fit(self.y_train_[ind].reshape(-1, 1))
                for ind in ind_X
            ]
        return kernel_density_list
        

def compute_individual_predictions(y_true, cde_preds, n_jobs = -1):
    def _compute_individual_density(y_, cde_pred):
        return np.exp(cde_pred.score([[y_]]))
    individual_pred = Parallel(n_jobs = n_jobs)(delayed(_compute_individual_density)(y_, cde_pred) 
                                                for y_, cde_pred in zip(y_true, cde_preds))
    return individual_pred
    

    
def phi_func(x, pilot_dict, kernel_type = "sinc", use_kernel = 1, dr = 0):
    T = pilot_dict['T']
    y = pilot_dict['y']
    def sinc_kernel(x):
        return np.sinc(x / np.pi)
    def triangular_kernel(x):
        return np.maximum(1 - np.abs(x), 0)
    if kernel_type == "sinc":
        density_estimator = sinc_kernel(T - T[:, None])
        density_pred = np.sum(density_estimator, axis=1) / len(T)
    elif kernel_type == "triangular":
        density_estimator = triangular_kernel(T - T[:, None])
        density_pred = np.sum(density_estimator, axis=1) / len(T)
    elif kernel_type == "gaussian":
        density_estimator = gaussian_kde(T)
        density_pred = density_estimator(T)
    
    complex_exponential = np.exp(1j * T * x)
    
    if dr == 1:
        T_reshaped = T.reshape(-1, 1)  # Reshape T to be a 2D array for sklearn
        y_reshaped = y.reshape(-1, 1)  # Reshape y to be a 2D array for sklearn
        model_Y_T = LinearRegression().fit(T_reshaped, y_reshaped)
        def integrand(t, x):
            prediction = model_Y_T.predict(np.array([[t]]))[0][0]
            # Compute the complex exponential
            complex_exponential_t = np.exp(1j * t * x)
        # Return the product of prediction and the complex exponential
            return prediction * complex_exponential_t
        t_points = np.linspace(40, 3000, 1000)  
        integrand_values = np.array([integrand(t, x) for t in t_points])
        integral_result = simpson(y=integrand_values, x=t_points)

        model_Y_T_values = model_Y_T.predict(T.reshape(-1, 1)).flatten()
        mean_result = integral_result + np.mean((y - model_Y_T_values)/ density_pred * complex_exponential)
    else:
        if use_kernel == 1:
            result = y / density_pred * complex_exponential
        else:
            result = y * complex_exponential
        mean_result = np.mean(result)
    real = np.real(mean_result)
    imag = np.imag(mean_result)
    return np.sqrt(real**2 + imag**2)

def est_A_r(pilot_dict, bnd, kernel_type = "sinc", use_kernel = 0, dr = 0):
    lb, ub = bnd
    # Create a new model
    model = gp.Model("MyModel")

    # Add variables
    A_var = model.addVar(name="A", vtype=gp.GRB.CONTINUOUS, lb=0.001, ub =1e+8)
    r = model.addVar(name="r", vtype=gp.GRB.INTEGER, lb=1, ub = 10)
    logA = model.addVar()

    # Set objective
    model.setObjective(logA * (np.log(ub)-np.log(lb)) - (r + 1) / 2 * (np.log(ub)**2 - np.log(lb)**2), gp.GRB.MINIMIZE)
    x_values = np.linspace(lb, ub, 10)  # Avoid x = 0 to prevent log(0)
    # Add constraint
    for x in x_values:
        model.addConstr(logA - (r + 1) * np.log(np.abs(x)) - np.log(phi_func(x, pilot_dict, kernel_type, use_kernel, dr)) >= 0)

    model.addGenConstrLog(A_var, logA)
    # Optimize model
    model.optimize()

    # Print results
    if model.status == gp.GRB.OPTIMAL:
        A_hat = A_var.X
        r_hat = r.X
        print(f"Optimal value of A: {A_var.X}")
        print(f"Optimal value of r (integer): {r.X}")
    else:
        print("No optimal solution found.")
    return (A_hat, r_hat)

def est_bias_bound(A_hat, r_hat, h, bnd_x, bnd_t):
    lb, ub = bnd_x
    t_lo, t_hi = bnd_t
    
    def integrand(t, x):
        prediction = np.sinc(t / np.pi)
        # Compute the complex exponential
        complex_exponential_t = np.exp(1j * t * x)
    # Return the product of prediction and the complex exponential
        return prediction * complex_exponential_t
    def fourier_transform(x):
        t_points = np.linspace(t_lo, t_hi, 1000)  
        integrand_values = np.array([integrand(t, x) for t in t_points])
        integral_result = simpson(y=integrand_values, x=t_points)
        return integral_result
    
    x_points = np.linspace(lb, ub, 200)
    integrand_values = np.array([np.abs(1-fourier_transform(h*x)) * A_hat * np.abs(x)**(-(r_hat+1)) for x in x_points])
    bias = 1/(2 * np.pi) * simpson(y=integrand_values, x=x_points)
    return bias


def est_model(data, num_fold = 2):
    X = data["x"]
    T = data["T"]
    y = data["y"]
    cv = KFold(n_splits=num_fold, shuffle=True, random_state=0)
    f_score_est = np.zeros(len(y))

    
    model_y_dict = {}
    cv_idx_dic_train = {}
    for i, (train_index, test_index) in enumerate(cv.split(X,T,y)):
        model_y = LinearRegression()
        cv_idx_dic_train[i] = test_index
        X_tr1, X_tr0, T_tr1, T_tr0, y_tr1, y_tr0 = X[train_index], X[test_index], T[train_index], T[test_index], y[train_index], y[test_index]
        ckde = ConditionalNearestNeighborsKDE().fit(X_tr1, T_tr1)
        ckde_preds = ckde.predict(X_tr0)
        f_score_est[test_index] = compute_individual_predictions(T_tr0, ckde_preds)
        X_full_tr = np.column_stack((X_tr1, T_tr1))
        X_full_tr = pd.DataFrame(X_full_tr)
        model_y_train = model_y.fit(X_full_tr, y_tr1)
        
        model_y_dict[i] = model_y_train
    f_score_est = np.maximum(f_score_est, 0.008)
    return (f_score_est, model_y_dict, cv_idx_dic_train)
    


def optimization_function(params, data_estimation):
    K, h = params
    y_train = data_estimation['y']
    T_train = data_estimation['T']
    X_train = data_estimation['x']
    X_policy_train = data_estimation['x_pol']
    cost = data_estimation['cost']
    t_lo = data_estimation['t_lo']
    t_hi = data_estimation['t_hi']
    f_score_est_train, model_y_train_dict, cv_idx_dic_train = data_estimation['f_score_est'], data_estimation['model_y_dict'], data_estimation['cv_idx_dic']
    N, P1 = X_policy_train.shape
    num_fold = len(model_y_train_dict)


    phi = np.zeros((N, K+1, P1))

    x_max = np.max(X_policy_train, axis = 0)
    x_min = np.min(X_policy_train, axis = 0)
    x_scale = x_max - x_min
    for k in range(K+1):
        phi[:,k,:] = (x_scale - np.abs(K*(X_policy_train - x_min)- k*x_scale)) * ((k - 1)*x_scale <= K*(X_policy_train - x_min)) * ((k+1)*x_scale >= K*(X_policy_train - x_min))

    # print("Data Diagnostics:")
    # print(f"X_train shape: {X_train.shape}")
    # print(f"T_train shape: {T_train.shape}")
    # print(f"y_train shape: {y_train.shape}")
    # print(f"y_train range: [{np.min(y_train):.2f}, {np.max(y_train):.2f}]")
    # print(f"T_train range: [{np.min(T_train):.2f}, {np.max(T_train):.2f}]")

    # Modify the optimization model
    m = gp.Model()
    m.setParam('Method', 2)
    m.setParam('NonConvex', 2)

    # Variables with more relaxed bounds
    beta = m.addVars(K+1, P1, vtype=gp.GRB.CONTINUOUS, lb=-100, ub=100, name="beta")
    pi = m.addVars(N, vtype=gp.GRB.CONTINUOUS, lb=t_lo, ub=t_hi, name="pi")
    slack = m.addVars(N, lb=-10000, ub=10000, name="slack")

    # Policy constraints with slack
    m.addConstrs(
        (pi[i] == gp.quicksum(beta[k, p]*phi[i,k,p] for p in range(P1) for k in range(K+1)) + slack[i]
        for i in range(N)),
        name="policy_constraints"
    )

    # Kernel terms
    kernel_term = m.addVars(N, lb=0, ub=1, name="kernel")
    abs_diff = m.addVars(N, lb=0, name="abs_diff")

    # Absolute value constraints
    m.addConstrs((abs_diff[i] >= (pi[i]-T_train[i])/h for i in range(N)))
    m.addConstrs((abs_diff[i] >= -(pi[i]-T_train[i])/h for i in range(N)))

    # Kernel constraints
    m.addConstrs((kernel_term[i] >= 1 - abs_diff[i] for i in range(N)))
    m.addConstrs((kernel_term[i] >= 0 for i in range(N)))
    m.addConstrs((kernel_term[i] <= 1 - abs_diff[i] for i in range(N)))
    m.addConstrs((beta[k, p] >= beta[k+1, p] for k in range(K) for p in range(P1)))
    # Simplified prediction variables
    y_pred = m.addVars(N, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="y_pred")

    # Add linear regression constraints for each fold
    for fold in range(num_fold):
        model_y_tmp = model_y_train_dict[fold]
        fold_idx = cv_idx_dic_train[fold]
        
        coef = model_y_tmp.coef_
        intercept = model_y_tmp.intercept_
        
        for i in fold_idx:
            m.addConstr(
                y_pred[i] == 
                gp.quicksum(coef[j] * X_train[i,j] for j in range(X_train.shape[1])) + 
                coef[-1] * pi[i] + 
                intercept
            )

    # Simplified objective
    obj = (
        gp.quicksum(kernel_term[i] * (y_train[i] - y_pred[i])/(h*f_score_est_train[i] + 1e-6) + y_pred[i] - cost * pi[i]
                    for i in range(N))/N -
        1e-4 * gp.quicksum(slack[i]*slack[i] for i in range(N)) -
        1e-6 * gp.quicksum(beta[k,p]*beta[k,p] for k in range(K+1) for p in range(P1))
    )

    m.setObjective(obj, gp.GRB.MAXIMIZE)

    # Solver parameters
    m.Params.NumericFocus = 3
    m.Params.TimeLimit = 600
    m.Params.BarConvTol = 1e-7
    m.Params.OptimalityTol = 1e-5
    m.Params.FeasibilityTol = 1e-5
    m.Params.Presolve = 2
    m.Params.MIPGap = 0.02


    try:
        m.optimize()
        print(f"\nOptimization Status: {m.status}")
        if m.status == gp.GRB.OPTIMAL:
            print(f"Objective Value: {m.objVal}")
            print(f"Max Slack: {max(abs(s.X) for s in slack.values())}")
    except gp.GurobiError as e:
        print(f"Error: {e}")

    # Save beta values and objective value in a dictionary
    results = {}

    if m.status == gp.GRB.OPTIMAL:
        # Get beta values
        beta_values = {}
        for k in range(K+1):
            for p in range(P1):
                beta_values[(k,p)] = beta[k,p].X
        
        results['beta'] = beta_values
        results['objective'] = m.objVal
    else:
        results['beta'] = None
        results['objective'] = None
    return results

    


def evaluation_function(results, params, data_testing):
    K, h = params
    X_test = data_testing['x']
    T_test = data_testing['T']
    y_test = data_testing['y']
    X_policy_test = data_testing['x_pol']
    cost = data_testing['cost']
    A_hat = data_testing['A_hat']
    r_hat = data_testing['r_hat']
    N_test, P1 = X_policy_test.shape
    f_score_est_test = data_testing['f_score_est']
    model_y_test_dict = data_testing['model_y_dict']
    cv_idx_dic_test = data_testing['cv_idx_dic']
    num_fold = len(model_y_test_dict)


    x_max = np.max(X_policy_test, axis = 0)
    x_min = np.min(X_policy_test, axis = 0)
    x_scale = x_max - x_min
    # Calculate phi values for test data (assuming same basis functions as training)
    phi_test = np.zeros((N_test, K+1, P1))
    for k in range(K+1):
        for p in range(P1):
            phi_test[:,k,p] = (x_scale[p] - np.abs(K*(X_policy_test[:,p] - x_min[p])- k*x_scale[p])) * ((k - 1)*x_scale[p] <= K*(X_policy_test[:,p] - x_min[p])) * ((k+1)*x_scale[p] >= K*(X_policy_test[:,p] - x_min[p]))


    # Calculate pi (policy values) for test data using optimal beta
    pi_test = np.zeros(N_test)
    for i in range(N_test):
        pi_test[i] = sum(results['beta'][(k,p)] * phi_test[i,k,p] 
                        for k in range(K+1) for p in range(P1))

    # Calculate kernel terms
    abs_diff_test = np.abs(pi_test - T_test)/h
    kernel_term_test = np.maximum(0, 1 - abs_diff_test)

    # Get predictions for test data using the same models as training
    y_pred_test = np.zeros(N_test)
    for fold in range(num_fold):
        model_y_tmp = model_y_test_dict[fold]
        fold_idx = cv_idx_dic_test[fold]
        y_pred_test[fold_idx] = model_y_tmp.predict(np.column_stack([X_test[fold_idx], pi_test[fold_idx]]))

    # Calculate objective value on test data
    obj_test = np.mean(kernel_term_test * (y_test - y_pred_test)/(h*f_score_est_test) + y_pred_test - cost * pi_test)

    bnd_t = (-100, 100)
    bnd_x = (1, 100)

    tau = np.sqrt(np.log(K)/(N_test*h))
    bias = est_bias_bound(A_hat, r_hat, h, bnd_x, bnd_t)
    penal_obj = obj_test - tau - 1.1 * bias


    return obj_test, penal_obj

    # print(f"\nTest Objective Value: {obj_test}")

def optimization_for_rad(params, data):
    K, h = params
    y_train = data['y']
    T_train = data['T']
    X_train = data['x']
    X_policy_train = data['x_pol']
    cost = data['cost']
    t_lo = data['t_lo']
    t_hi = data['t_hi']
    f_score_est_train, model_y = data['f_score_est'], data['model_y']
    N, P1 = X_policy_train.shape


    phi = np.zeros((N, K+1, P1))

    x_max = np.max(X_policy_train, axis = 0)
    x_min = np.min(X_policy_train, axis = 0)
    x_scale = x_max - x_min
    for k in range(K+1):
        phi[:,k,:] = (x_scale - np.abs(K*(X_policy_train - x_min)- k*x_scale)) * ((k - 1)*x_scale <= K*(X_policy_train - x_min)) * ((k+1)*x_scale >= K*(X_policy_train - x_min))

    # print("Data Diagnostics:")
    # print(f"X_train shape: {X_train.shape}")
    # print(f"T_train shape: {T_train.shape}")
    # print(f"y_train shape: {y_train.shape}")
    # print(f"y_train range: [{np.min(y_train):.2f}, {np.max(y_train):.2f}]")
    # print(f"T_train range: [{np.min(T_train):.2f}, {np.max(T_train):.2f}]")

    # Modify the optimization model
    m = gp.Model()
    m.setParam('Method', 2)
    m.setParam('NonConvex', 2)

    # Variables with more relaxed bounds
    beta = m.addVars(K+1, P1, vtype=gp.GRB.CONTINUOUS, lb=-100, ub=100, name="beta")
    pi = m.addVars(N, vtype=gp.GRB.CONTINUOUS, lb=t_lo, ub=t_hi, name="pi")
    slack = m.addVars(N, lb=-10000, ub=10000, name="slack")

    # Policy constraints with slack
    m.addConstrs(
        (pi[i] == gp.quicksum(beta[k, p]*phi[i,k,p] for p in range(P1) for k in range(K+1)) + slack[i]
        for i in range(N)),
        name="policy_constraints"
    )

    # Kernel terms
    kernel_term = m.addVars(N, lb=0, ub=1, name="kernel")
    abs_diff = m.addVars(N, lb=0, name="abs_diff")

    # Absolute value constraints
    m.addConstrs((abs_diff[i] >= (pi[i]-T_train[i])/h for i in range(N)))
    m.addConstrs((abs_diff[i] >= -(pi[i]-T_train[i])/h for i in range(N)))

    # Kernel constraints
    m.addConstrs((kernel_term[i] >= 1 - abs_diff[i] for i in range(N)))
    m.addConstrs((kernel_term[i] >= 0 for i in range(N)))
    m.addConstrs((kernel_term[i] <= 1 - abs_diff[i] for i in range(N)))
    m.addConstrs((beta[k, p] >= beta[k+1, p] for k in range(K) for p in range(P1)))
    # Simplified prediction variables
    y_pred = m.addVars(N, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="y_pred")

    # Add linear regression constraints for each fold
    

        
    coef = model_y.coef_
    intercept = model_y.intercept_
    
    for i in range(N):
        m.addConstr(
            y_pred[i] == 
            gp.quicksum(coef[j] * X_train[i,j] for j in range(X_train.shape[1])) + 
            coef[-1] * pi[i] + 
            intercept
        )

    # Simplified objective
    obj = (
        gp.quicksum(kernel_term[i] * (y_train[i] - y_pred[i])/(h*f_score_est_train[i] + 1e-6) + y_pred[i] - cost * pi[i]
                    for i in range(N))/N -
        1e-4 * gp.quicksum(slack[i]*slack[i] for i in range(N)) -
        1e-6 * gp.quicksum(beta[k,p]*beta[k,p] for k in range(K+1) for p in range(P1))
    )

    m.setObjective(obj, gp.GRB.MAXIMIZE)

    # Solver parameters
    m.Params.NumericFocus = 3
    m.Params.TimeLimit = 600
    m.Params.BarConvTol = 1e-7
    m.Params.OptimalityTol = 1e-5
    m.Params.FeasibilityTol = 1e-5
    m.Params.Presolve = 2
    m.Params.MIPGap = 0.02


    try:
        m.optimize()
        print(f"\nOptimization Status: {m.status}")
        if m.status == gp.GRB.OPTIMAL:
            print(f"Objective Value: {m.objVal}")
            print(f"Max Slack: {max(abs(s.X) for s in slack.values())}")
    except gp.GurobiError as e:
        print(f"Error: {e}")

    # Save beta values and objective value in a dictionary
    results = {}

    if m.status == gp.GRB.OPTIMAL:
        # Get beta values
        beta_values = {}
        for k in range(K+1):
            for p in range(P1):
                beta_values[(k,p)] = beta[k,p].X
        
        results['beta'] = beta_values
        results['objective'] = m.objVal
    else:
        results['beta'] = None
        results['objective'] = None
    return results




def cal_rademacher(params, input_data):
    y = input_data['y']
    T = input_data['T']
    X = input_data['x']
    t_lo = input_data['t_lo']
    t_hi = input_data['t_hi']
    X_policy = input_data['x_pol']
    cost = input_data['cost']
    f_score_est = input_data['f_score_est']
    model_y_dict = input_data['model_y_dict']
    cv_idx_dic = input_data['cv_idx_dic']
    num_fold = len(model_y_dict)
    rad_dic = {}
    for fold in range(num_fold):
        rad_val_list = []
        model_y_tmp = model_y_dict[fold]
        fold_idx = cv_idx_dic[fold]
        y_tmp = y[fold_idx]
        T_tmp = T[fold_idx]
        X_tmp = X[fold_idx]
        X_policy_tmp = X_policy[fold_idx]
        for sim in range(100): 
            sigma = np.random.choice([-1, 1], size=len(y_tmp), p=[0.5, 0.5])
            y_tmp_sim = 2 * y_tmp * sigma
            data_tmp = {'y': y_tmp_sim, 'T': T_tmp, 'x': X_tmp, 'x_pol': X_policy_tmp, 'cost': cost, 't_lo': t_lo, 't_hi': t_hi, 'f_score_est': f_score_est, 'model_y': model_y_tmp, 'cv_idx_dic': fold_idx}
            results = optimization_for_rad(params, data_tmp)
            if results['objective'] is not None:
                rad_val_list.append(results['objective'])
        rad_dic[fold] = np.mean(rad_val_list) * len(y_tmp) / len(y)
    rad_dic_avg = np.mean(list(rad_dic.values()))
    return rad_dic_avg
        
        
def evaluation_function_rad(params, input_data):
    K, h = params
    results = optimization_function(params, input_data)
    obj = results['objective']
    beta = results['beta']
    rad_avg = cal_rademacher(params, input_data)
    bnd_t = (-100, 100)
    bnd_x = (1, 100)

    tau = np.sqrt(np.log(K)/(len(input_data['y'])*h))
    bias = est_bias_bound(input_data['A_hat'], input_data['r_hat'], h, bnd_x, bnd_t)
    penal_obj = obj - rad_avg - tau - 1.1 * bias
    return obj, penal_obj, beta
