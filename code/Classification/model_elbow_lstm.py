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
from matplotlib import pyplot as plt
import seaborn as sns

# Constants
#PATH_TO_RAW_CSV = './raw_csv_elbow'  # Path to raw csv data
PATH_TO_RAW_CSV = './Oneuser_data'  # Path to raw csv data
#PATH_TO_RAW_CSV = './raw_csv_20'  # Path to raw csv data
PATH_TO_EXTRACTED = './extracted_data/'  # Path where extracted features will be saved
SWING_TYPES = ['bending45', 'bending90', 'ExterRota','ExterTwist','InterRota','InterTwist'] # List of the possible swing types
SWING_TO_INT = {'bending45': 0, 'bending90': 1, 'ExterRota': 2, 'ExterTwist': 3, 'InterRota': 4, 'InterTwist': 5} # Mapping of swing type to integer representation
AXES = ['x', 'y', 'z']  # Three axes that measurements come in
NUM_END = 1  # Case where final data entry is incomplete
Number = ['1', '2', '3'] # Three axes that measurements come in
training_list = []

def evalute_Fold_dl(model, fold, X, y):
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
        accuracy = accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1),)
        precision = precision_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='weighted')
        recall = recall_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='weighted')
        f1 = f1_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='weighted')
        ACCscores.append(accuracy)
        PREscores.append(precision)
        REscores.append(recall)
        F1scores.append(f1)
    return np.mean(ACCscores), np.mean(PREscores), np.mean(REscores), np.mean(F1scores), np.std(ACCscores), np.std(PREscores), np.std(REscores), np.std(F1scores)


def get_data():
    # Load training data into numpy array
    data = []
    label = []
    for swing_type in SWING_TYPES:
        swing_list = []
        # Adds the path to every swing of the swing type into a list
        for root, dir, files in os.walk(os.path.join(PATH_TO_RAW_CSV, swing_type)):
            for name in files:
                swing_list.append(os.path.join(root, name))
        # For every swing of this type, extract the features and append to the training list
        for swing_path in swing_list:
            df = pd.read_csv(swing_path)
            #df1 = df.iloc[:, 0:9]           ############################ all data 9
            #df1 = df.iloc[:, 0:3]           ########################### only gre  3
            df1 = df.iloc[:, 3:9]           ########################### only imu  6
            label.append(SWING_TO_INT[swing_type])
            data.append(df1)
    return data, label

def dlPreprocess():
    input_array = []
    classes = []
    input_array, classes = get_data()
    max_len = max([len(i) for i in input_array])
    for i in input_array:
        if (len(i) == max_len):
            largest_array = i
    padded_array = []
    for i in input_array:
        # zero numpy array
        zero_array = np.zeros(largest_array.shape)
        zero_array[:i.shape[0], :i.shape[1]] = i
        padded_array.append(zero_array)
    # convert to 3d numpy array
    padded_array = np.dstack(padded_array)
    # transpose array
    padded_array = padded_array.transpose(2, 0, 1)
    #ic(padded_array.shape)
    #ic(classes.shape)
    return padded_array, classes

def create_segments_and_labels(df, label, windowsize, stridewindow):
    segments = []
    labels = []
    for j in range(len(df)):
        col = len(df[j])
        length = int(np.floor((col-windowsize)/(windowsize-stridewindow))+1)
        for i in range(length):
            temp = df[j][i*windowsize: (i+1)*windowsize, :]
            segments.append(temp)
            labels.append(label[j])
    lenFeature = segments[0].shape[1]
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, windowsize, lenFeature)
    labels = np.asarray(labels)
    return reshaped_segments, labels

LABELS=[]
for i in range(1,7,1):
  LABELS.append (i)
print(LABELS)

def show_confusion_matrix(validations, predictions):
    matrix = confusion_matrix(validations, predictions)
    plt.figure(figsize=(20, 14))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def dataOneHot(data):
    values = np.array(data)
    #ic(values.shape)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    #ic(len(integer_encoded))
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

def lstm_train():
    # read data ###########################################
    input_array, output_array = dlPreprocess()
    ######################################################
    ic(input_array.shape)
    #ic(len(output_array))
    X_data, y_data = create_segments_and_labels(input_array, output_array, 225, 0)
    ic(X_data.shape)
    ic(y_data.shape)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=0)
    yTrainOneHot = dataOneHot(y_train)
    yTestOneHot = dataOneHot(y_test)
    # Retrieve from input array shape
    time_steps = X_train.shape[1]
    n_features = X_train.shape[2]
    input_shape = (time_steps, n_features)

    n_steps, n_length, n_depth = 15, 15, 6   ######################################## n_depth need change

    X_train = X_train.reshape(X_train.shape[0], n_steps, n_length, n_depth)
    X_test = X_test.reshape(X_test.shape[0], n_steps, n_length, n_depth)
    ic(X_train.shape)
    n_outputs = yTrainOneHot.shape[1]
    #### lstm model
    verbose, epochs, batch_size = 0, 200, 64
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=32, kernel_size=2, padding='same', kernel_initializer="he_normal",strides=1,kernel_regularizer=l1(1e-04)), \
                               input_shape=(n_steps,n_length,n_depth)))
    model.add(TimeDistributed(BatchNormalization(epsilon=1e-06, momentum=0.95, weights=None)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=4,strides=2)))
    model.add(TimeDistributed(Activation('tanh')))
    model.add(TimeDistributed(Conv1D(filters=16, kernel_size=2,padding="same",kernel_initializer="he_normal",strides=1,kernel_regularizer=l1(1e-04))))
    model.add(TimeDistributed(BatchNormalization(epsilon=1e-06, momentum=0.95, weights=None)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=4,strides=2)))
    model.add(TimeDistributed(Activation('tanh')))
    model.add(TimeDistributed(Dropout(0.2093)))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Flatten()))
    # model.add(Flatten())
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='tanh'))
    model.add(BatchNormalization(epsilon=1e-05, momentum=0.9, weights=None))
    model.add(Dense(n_outputs, activation='softmax'))

    adam = optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, amsgrad=False)
    checkpoint_filepath = '/media/naveen/nav/mat_codes/nina_DB1_codes/nina_prep_naveen/CNN25X20/checkpoint.hdf5'
    # model.load_weights(checkpoint_filepath)
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, verbose=1, monitor='val_accuracy',
                                          save_weights_only=True, save_best_only=True)
    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='auto', baseline=None,
                          restore_best_weights=True)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    ic(model.summary())

    ## fit and save model
    history = model.fit(X_train, yTrainOneHot, epochs=epochs, batch_size=batch_size, validation_data=(X_test, yTestOneHot), verbose=1)
    ###########
    best_index = history.history['val_acc'].index(max(history.history['val_acc']))
    ########################################################################
    #np.save("./elbow_result/imu_only/training_acc.npy",history.history['acc'])
    #np.save("./elbow_result/imu_only/training_loss.npy", history.history['loss'])
    #np.save("./elbow_result/imu_only/validation_acc.npy",history.history['val_acc'] )
    #np.save("./elbow_result/imu_only/validation_loss.npy", history.history['val_loss'])

    #np.save("./elbow_result/Oneuser/imu_only/training_acc.npy",history.history['acc'])
    #np.save("./elbow_result/Oneuser/imu_only/training_loss.npy", history.history['loss'])
    #np.save("./elbow_result/Oneuser/imu_only/validation_acc.npy",history.history['val_acc'] )
    #np.save("./elbow_result/Oneuser/imu_only/validation_loss.npy", history.history['val_loss'])

    np.save("./elbow_result/Oneuser/imu_only/training_acc.npy",history.history['acc'])
    np.save("./elbow_result/Oneuser/imu_only/training_loss.npy", history.history['loss'])
    np.save("./elbow_result/Oneuser/imu_only/validation_acc.npy",history.history['val_acc'] )
    np.save("./elbow_result/Oneuser/imu_only/validation_loss.npy", history.history['val_loss'])
    #
    """
    np.save("./elbow_result/gre_only/training_acc.npy", history.history['acc'])
    np.save("./elbow_result/gre_only/training_loss.npy", history.history['loss'])
    np.save("./elbow_result/gre_only/validation_acc.npy", history.history['val_acc'] )
    np.save("./elbow_result/gre_only/validation_loss.npy", history.history['val_loss'])
    #
   """
    #np.save("./elbow_result/all/training_acc.npy", history.history['acc'])
    #np.save("./elbow_result/all/training_loss.npy", history.history['loss'])
    #np.save("./elbow_result/all/validation_acc.npy", history.history['val_acc'])
    #np.save("./elbow_result/all/validation_loss.npy", history.history['val_loss'])
    ########################################################################
    # print('epoch_number', best_index+1)
    # print('train accuracy and validation accuracy', history.history['acc'][best_index], history.history['val_acc'][best_index])

    ########################################################################
    #model.save('./elbow_result/imu_only/lstm.h5')
    #model.save('./elbow_result/gre_only/lstm.h5')
    #model.save('./elbow_result/all/lstm.h5')
    model.save('./elbow_result/Oneuser/imu_only/lstm.h5')
    ########################################################################

    y_pred_train = model.predict(X_train)
    max_y_pred_train = np.argmax(y_pred_train, axis=1)
    #show_confusion_matrix(y_train, max_y_pred_train)
    matrix_train = confusion_matrix(y_train, max_y_pred_train)

    ########################################################################
    #np.save("./elbow_result/imu_only/training_confusion_matrix.npy",matrix_train)
    #np.save("./elbow_result/gre_only/training_confusion_matrix.npy", matrix_train)
    #np.save("./elbow_result/all/training_confusion_matrix.npy", matrix_train)
    np.save("./elbow_result/Oneuser/imu_only/training_confusion_matrix.npy", matrix_train)
    ########################################################################

    # print(classification_report(y_train, max_y_pred_train))
    y_pred_test = model.predict(X_test)
    max_y_pred_test = np.argmax(y_pred_test, axis=1)
    max_y_test = np.argmax(yTestOneHot, axis=1)
    #show_confusion_matrix(max_y_test, max_y_pred_test)
    matrix_test = confusion_matrix(max_y_test, max_y_pred_test)

    ########################################################################
    #np.save("./elbow_result/imu_only/test_confusion_matrix.npy",matrix_test)
    #np.save("./elbow_result/gre_only/test_confusion_matrix.npy", matrix_test)
    #np.save("./elbow_result/all/test_confusion_matrix.npy", matrix_test)
    np.save("./elbow_result/Oneuser/imu_only/test_confusion_matrix.npy", matrix_test)
    ########################################################################


    lstmRFACC, lstmRFPRE, lstmRFRE, lstmRFF1, STDlstmRFACC, STDlstmRFPRE, STDlstmRFRE, STDlstmRFF1  = evalute_Fold_dl(model, 5, X_train, yTrainOneHot)
    print("lstmRF-ACC: %.4f" % (lstmRFACC))
    print("lstmRF-PRE: %.4f" % (lstmRFPRE))
    print("lstmRF-RE: %.4f" % (lstmRFRE))
    print("lstmRF-F1: %.4f" % (lstmRFF1))
    print("STDlstmRF-ACC: %.4f" % (STDlstmRFACC))
    print("STDlstmRF-PRE: %.4f" % (STDlstmRFPRE))
    print("STDlstmRF-RE: %.4f" % (STDlstmRFRE))
    print("STDlstmRF-F1: %.4f" % (STDlstmRFF1))


if __name__ == '__main__':
    lstm_train()


