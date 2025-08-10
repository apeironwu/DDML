

## loading package
import numpy as np
import pandas as pd
import random
import pickle
from sklearn.ensemble import RandomForestRegressor
from timeit import default_timer as timer
import time
import sys, getopt
import math
from sklearn.model_selection import GridSearchCV


## set path
import os
os.chdir('/lustre/project/Stat/s1155168529/programs/DDML/code/real_data')
os.getcwd()


## set random seed
np.random.seed(2024)
random.seed(2024)


## set parameter
n_iter = 1
K      = 3  #number of sites 
K_fold = 3
n_rft  = 50


## load data
vec_beta_est_iter = np.zeros(n_iter)

adni1_pd  = pd.read_csv('adni1.csv')
adni2_pd  = pd.read_csv('adni2.csv')
adnigo_pd = pd.read_csv('adnigo.csv')

adni1  = adni1_pd.values
adni2  = adni2_pd.values
adnigo = adnigo_pd.values

## concatenate data
df = [adni1, adni2, adnigo]


## nuisance parameter estimation
model = RandomForestRegressor()

param_grid = {
    'n_estimators'     : [20, 50, 100, 200],
    'max_depth'        : [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf' : [1, 5, 10]
}

# Setup the grid search
grid_search = GridSearchCV(
    estimator = model, 
    param_grid = param_grid, 
    cv = 3, 
    scoring = 'neg_mean_squared_error', 
    verbose = 1
)

# Fit grid search
best_model1_d = grid_search.fit(df[0][:,6:20], df[0][:,22])

# Get the best parameters
print("Best parameters:", best_model1_d.best_params_)


param_grid = {
    'n_estimators': [20, 50, 100],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10]
}

# Setup the grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=4, scoring='neg_mean_squared_error', verbose=1)

# Fit grid search
best_model1_y = grid_search.fit(df[0][:,6:20], df[0][:,21])

# Get the best parameters
print("Best parameters:", best_model1_y.best_params_)



param_grid = {
    'n_estimators': [20, 50, 100],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10]
}

# Setup the grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)

# Fit grid search
best_model2_d = grid_search.fit(df[1][:,6:20], df[1][:,22])

# Get the best parameters
print("Best parameters:", best_model2_d.best_params_)


param_grid = {
    'n_estimators': [20, 50, 100],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10]
}

# Setup the grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)

# Fit grid search
best_model2_y = grid_search.fit(df[1][:,6:20], df[1][:,21])

# Get the best parameters
print("Best parameters:", best_model2_y.best_params_)



param_grid = {
    'n_estimators': [20, 50, 100],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10]
}

# Setup the grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)

# Fit grid search
best_model3_d = grid_search.fit(df[2][:,6:20], df[2][:,22])

# Get the best parameters
print("Best parameters:", best_model3_d.best_params_)



param_grid = {
    'n_estimators': [20, 50, 100],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10]
}

# Setup the grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)

# Fit grid search
best_model3_y = grid_search.fit(df[2][:,6:20], df[2][:,21])

# Get the best parameters
print("Best parameters:", best_model3_y.best_params_)


## concatenate 
param_d = [
    best_model1_d.best_params_, 
    best_model2_d.best_params_, 
    best_model3_d.best_params_
]
param_y = [
    best_model1_y.best_params_,
    best_model2_y.best_params_, 
    best_model3_y.best_params_
]

pickle.dump(
   [param_d, param_y], 
   open("rf_set.pydata", "wb")
)











