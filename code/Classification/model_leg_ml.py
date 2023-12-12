import os
import pandas as pd
import numpy as np
from sklearn import model_selection, linear_model, neighbors, ensemble
from icecream import ic
import DataFeature
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv1D,Conv2D, MaxPooling1D,AveragePooling1D,Activation
from tensorflow.keras.regularizers import l2,l1
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape
from tensorflow.keras import optimizers
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential,Model, load_model
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, zero_one_loss, confusion_matrix, mean_squared_error
from sklearn import svm

# Constants
PATH_TO_RAW_CSV = './raw_csv'  # Path to raw csv data
PATH_TO_EXTRACTED = './extracted_data/'  # Path where extracted features will be saved
SWING_TYPES = ['Half_squat', 'HC_45', 'HC_90', 'Leg_extension_sit',
               'Leg_presses_liftup']  # List of the possible swing types
SWING_TO_INT = {'Half_squat': 0, 'HC_45': 1, 'HC_90': 2, 'Leg_extension_sit': 3,
                'Leg_presses_liftup': 4}  # Mapping of swing type to integer representation
AXES = ['x', 'y', 'z']  # Three axes that measurements come in
NUM_END = 1  # Case where final data entry is incomplete
Number = ['1', '2', '3', '4', '5'] # Three axes that measurements come in
training_list = []

def extract_features_all(df, label):
    df = df.drop(df.tail(NUM_END).index)
    extracted = []
    extracted.append(SWING_TO_INT[label])
    for direction in Number:
        col = 'S{}'.format(direction)
        data = df[[col]]
        arr = data.to_numpy()
        #extracted.append(DataFeature.TD_mean(arr))
        extracted.append(DataFeature.TD_std(arr))
        extracted.append(DataFeature.TD_mav(arr))
        #extracted.append(DataFeature.TD_max(arr))
        #extracted.append(DataFeature.TD_min(arr))

    for direction in AXES:
        col = 'a{}'.format(direction)
        data = df[[col]]
        arr = data.to_numpy()
        #extracted.append(DataFeature.TD_mean(arr))
        extracted.append(DataFeature.TD_std(arr))
        extracted.append(DataFeature.TD_mav(arr))
        #extracted.append(DataFeature.TD_max(arr))
        #extracted.append(DataFeature.TD_min(arr))

    for direction in AXES:
        col = 'g{}'.format(direction)
        data = df[[col]]
        arr = data.to_numpy()
        #extracted.append(DataFeature.TD_mean(arr))
        extracted.append(DataFeature.TD_std(arr))
        extracted.append(DataFeature.TD_mav(arr))
        #extracted.append(DataFeature.TD_max(arr))
        #extracted.append(DataFeature.TD_min(arr))

    return extracted

def extract_features_imu(df, label):
    df = df.drop(df.tail(NUM_END).index)
    extracted = []
    extracted.append(SWING_TO_INT[label])
    for direction in AXES:
        col = 'a{}'.format(direction)
        data = df[[col]]
        arr = data.to_numpy()
        #extracted.append(DataFeature.TD_mean(arr))
        extracted.append(DataFeature.TD_std(arr))
        extracted.append(DataFeature.TD_mav(arr))
        #extracted.append(DataFeature.TD_max(arr))
        #extracted.append(DataFeature.TD_min(arr))

    for direction in AXES:
        col = 'g{}'.format(direction)
        data = df[[col]]
        arr = data.to_numpy()
        #extracted.append(DataFeature.TD_mean(arr))
        extracted.append(DataFeature.TD_std(arr))
        extracted.append(DataFeature.TD_mav(arr))
        #extracted.append(DataFeature.TD_max(arr))
        #extracted.append(DataFeature.TD_min(arr))

    return extracted

def extract_features_gre(df, label):
    df = df.drop(df.tail(NUM_END).index)
    extracted = []
    extracted.append(SWING_TO_INT[label])
    for direction in Number:
        col = 'S{}'.format(direction)
        data = df[[col]]
        arr = data.to_numpy()
        #extracted.append(DataFeature.TD_mean(arr))
        #extracted.append(DataFeature.TD_std(arr))
        #extracted.append(DataFeature.TD_mav(arr))
        extracted.append(DataFeature.TD_max(arr))
        extracted.append(DataFeature.TD_min(arr))

    return extracted

def get_training_data():
    for swing_type in SWING_TYPES:
        swing_list = []
        for root, dir, files in os.walk(os.path.join(PATH_TO_RAW_CSV, swing_type)):
            for name in files:
                swing_list.append(os.path.join(root, name))
        for swing_path in swing_list:
            df = pd.read_csv(swing_path)
            extracted_swing = extract_features_imu(df, swing_type)
            training_list.append(extracted_swing)
    training_df = pd.DataFrame(data=training_list).to_numpy()
    ic(training_df.shape)
    # Separate array into labels and data
    y = training_df[:,0]
    X = training_df[:,1:]
    return X, y

def evalute_Fold(model, fold, X, y):
    X = np.array(X)
    y = np.array(y)
    kf = KFold(n_splits=fold, shuffle=True)
    ACCscores = []
    PREscores = []
    REscores = []
    F1scores = []
    for k, (train, test) in enumerate(kf.split(X, y)):
        X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        ACCscores.append(accuracy)
        PREscores.append(precision)
        REscores.append(recall)
        F1scores.append(f1)
    return np.mean(ACCscores), np.std(ACCscores), np.mean(PREscores), np.std(PREscores), np.mean(REscores), np.std(REscores), np.mean(F1scores), np.std(F1scores)

def model_train():
    # get data
    X, y = get_training_data()
    #############################################################
    ##method svm
    SVM = svm.SVC(C=1.0, kernel='rbf')
    SVMACC, STDSVMACC, SVMPRE, STDSVMPRE, SVMRE, STDSVMRE, SVMF1, STDSVMF1 = evalute_Fold(SVM, 5, X, y)
    print("SVM-ACC: %.4f" % (SVMACC))
    print("STDSVM-ACC: %.4f" % (STDSVMACC))
    print("SVM-PRE: %.4f" % (SVMPRE))
    print("STDSVM-PRE: %.4f" % (STDSVMPRE))
    print("SVM-RE: %.4f" % (SVMRE))
    print("STDSVM-RE: %.4f" % (STDSVMRE))
    print("SVM-F1: %.4f" % (SVMF1))
    print("STDSVM-F1: %.4f" % (STDSVMF1))
    print("##########")
    ##method rf
    RF = ensemble.RandomForestClassifier()
    RFACC, STDRFACC, RFPRE,STDRFPRE, RFRE, STDRFRE,  RFF1, STDRFF1 = evalute_Fold(RF, 5, X, y)
    print("RFACC: %.4f" % (RFACC))
    print("STDRFACC: %.4f" % (STDRFACC))
    print("RF-PRE: %.4f" % (RFPRE))
    print("STDRF-PRE: %.4f" % (STDRFPRE))
    print("RF-RE: %.4f" % (RFRE))
    print("STDRF-RE: %.4f" % (STDRFRE))
    print("RFF1: %.4f" % (RFF1))
    print("STDRFF1: %.4f" % (STDRFF1))
    print("##########")

if __name__ == '__main__':
    #train_1()
    model_train()



