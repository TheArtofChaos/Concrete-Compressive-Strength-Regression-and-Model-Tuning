#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 11:16:57 2023

@author: najeh
"""
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, MaxAbsScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np

# Load the data
data = pd.read_excel('/Users/najeh/Downloads/Concrete_Data.xls')
X = data.drop(columns=['Concrete compressive strength(MPa, megapascals)'])
y = data['Concrete compressive strength(MPa, megapascals)']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define scalers
scalers = [StandardScaler(), MinMaxScaler(), RobustScaler(), QuantileTransformer(), MaxAbsScaler()]

# Define hyperparameters for tuning
rf_params = {'n_estimators': [100, 200, 300], 'max_features': ['auto', 'sqrt'], 'bootstrap': [True, False]}
dt_params = {'max_depth': [10, 20, 30, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}

# Initiate the regressors
rf_reg = RandomForestRegressor(random_state=42)
dt_reg = DecisionTreeRegressor(random_state=42)

# Grid search cross-validation
for scaler in scalers:
    print('Scaler:', scaler)
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf_grid_search = GridSearchCV(rf_reg, rf_params, cv=5, scoring='neg_mean_squared_error')
    rf_grid_search.fit(X_train_scaled, y_train)
    
    rf_pred = rf_grid_search.predict(X_test_scaled)
    print('Random Forest RMSE:', np.sqrt(mean_squared_error(y_test, rf_pred)))
    print('Random Forest MAE:', mean_absolute_error(y_test, rf_pred))
    print('Random Forest R2:', r2_score(y_test, rf_pred))
    print('Best RF Parameters:', rf_grid_search.best_params_)
    
    dt_grid_search = GridSearchCV(dt_reg, dt_params, cv=5, scoring='neg_mean_squared_error')
    dt_grid_search.fit(X_train_scaled, y_train)
    
    dt_pred = dt_grid_search.predict(X_test_scaled)
    print('Decision Tree RMSE:', np.sqrt(mean_squared_error(y_test, dt_pred)))
    print('Decision Tree MAE:', mean_absolute_error(y_test, dt_pred))
    print('Decision Tree R2:', r2_score(y_test, dt_pred))
    print('Best DT Parameters:', dt_grid_search.best_params_)

