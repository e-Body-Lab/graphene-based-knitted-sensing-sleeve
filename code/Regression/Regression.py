import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import (LinearRegression, HuberRegressor,  RANSACRegressor, TheilSenRegressor)
from sklearn.decomposition import PCA
from pathlib import Path
import glob
import os
import scipy.ndimage as ndimage
from scipy import signal
import statsmodels.api as smapi
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from icecream import ic
from sklearn.ensemble import RandomForestRegressor    ###############  added

# Specify your data path here
#movements = 'HC_90'
#movements = 'HC_45'
#movements = 'Half_squat'
#movements = 'Leg_presses_liftup'
movements = 'sit'
#ic(movements)
data_dir = Path(r'./Formal_data_collection_regreeesion/' + movements)

#data_dir = Path(r'./Gene/Regression/train/')  ## 所有测试
#test_data_dir = Path(r'./Gene/Regression/test/')  ## 所有测试

def get_data():
    all_filenames = []
    for path, subdirs, files in os.walk(data_dir):
        for name in files:
            all_filenames.append(os.path.join(path, name))
    ic(all_filenames)
    tcs = pd.concat((pd.read_csv(f, index_col=False, header=None) for f in all_filenames))
    tcs = tcs.loc[:, :5]
    return tcs

def get_test_data():
    all_filenames = []
    for path, subdirs, files in os.walk(test_data_dir):
        for name in files:
            all_filenames.append(os.path.join(path, name))
    ic(all_filenames)
    tcs = pd.concat((pd.read_csv(f, index_col=False, header=None) for f in all_filenames))
    tcs = tcs.loc[:, :5]
    return tcs

def filter(data):
    X = data.iloc[:, 1: 6]  # Manually changing sensor number, all: 1-6; serson 1 : 1-2
    ic(X)
    y = data.iloc[:, 0]
    low_filt = True
    g_filt = False
    if low_filt:
        # tcs = np.array(tcs)
        # tcs_filt = ndimage.gaussian_filter(tcs,(0,3))
        # tcs_filt = pd.DataFrame(tcs_filt)
        sos = signal.butter(1, 1.5, 'low', fs=100, output='sos')
        X = signal.sosfilt(sos, X)
    if g_filt:
        X = np.array(X)
        X = ndimage.gaussian_filter(X, (0, 3))
    return X, y

def filter_each(data,step):
    X = data.iloc[:, step: step+1]  # Manually changing sensor number, all: 1-6; serson 1 : 1-2
    ic(X)
    y = data.iloc[:, 0]
    low_filt = True
    g_filt = False
    if low_filt:
        # tcs = np.array(tcs)
        # tcs_filt = ndimage.gaussian_filter(tcs,(0,3))
        # tcs_filt = pd.DataFrame(tcs_filt)
        sos = signal.butter(1, 1.5, 'low', fs=100, output='sos')
        X = signal.sosfilt(sos, X)
    if g_filt:
        X = np.array(X)
        X = ndimage.gaussian_filter(X, (0, 3))
    return X, y

def error(refer, estimate):
    MSE = mean_squared_error(refer, estimate)
    RMSE = np.sqrt(MSE)
    RSquared = 1 - mean_squared_error(refer, estimate)/np.var(refer)
    MAE = mean_absolute_error(refer, estimate)
    return RMSE, RSquared, MAE

def each_sensor_regression():
    tcs = get_data()
    for step in [1, 2, 3, 4, 5]:
        X, y = filter_each(tcs, step)
        ###
        LR = LinearRegression().fit(X, y)
        LR_pred = LR.predict(X)
        ##  Added section
        #model_RF = RandomForestRegressor(n_estimators=50)
        RF = RandomForestRegressor()
        RF.fit(X, y)
        RF_pred = RF.predict(X)
        ##
        #XGB = xgb.XGBRegressor()
        #XGB.fit(X, y)
        #XGB_pred = XGB.predict(X)
        ###
        #Svr = SVR(kernel='rbf')
        #Svr.fit(X, y)
        #Svr_pred = Svr.predict(X)
        #
        lr_rmse, lr_r2, lr_mae = error(y, LR_pred)
        rf_rmse, rf_r2 , rf_mae = error(y, RF_pred)

        ic(lr_rmse)
        ic(lr_r2)
        ic(lr_mae)
        ic(rf_rmse)
        ic(rf_r2)
        ic(rf_mae)

def all_regression():
    train_tcs = get_data()
    X_train, y_train = filter(train_tcs)
    test_tcs = get_test_data()
    X_test, y_test = filter(test_tcs)

    ###
    LR = LinearRegression().fit(X_train, y_train)
    LR_pred = LR.predict(X_test)
    ##
    #model_RF = RandomForestRegressor(n_estimators=50)
    RF = RandomForestRegressor()
    RF.fit(X_train, y_train)
    RF_pred = RF.predict(X_test)
    ###
    lr_rmse, lr_r2, lr_mae = error(y_test, LR_pred)
    rf_rmse, rf_r2, rf_mae = error(y_test, RF_pred)

    ic(lr_rmse)
    ic(lr_r2)
    ic(lr_mae)
    ic(rf_rmse)
    ic(rf_r2)
    ic(rf_mae)

if __name__ == '__main__':
    each_sensor_regression()
    #all_regression()