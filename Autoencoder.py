# Import the required Libraries
 
# Pandas for data manipulation and analysis. It offers data structures and operations for manipulating numerical tables and time series.
import pandas as pd    
 
# Numpy for adding support for high-level mathematical functions to operate on the large, multi-dimensional arrays and matrices.
import numpy as np     
 
# provides a way of using operating system dependent functionality with the underlying operating system that Python is running on.
import os              
 
#MinMaxScaler Transforms features by scaling each feature to a given range e.g. between zero and one.
from sklearn.preprocessing import MinMaxScaler
 
""" Shuffle method used to shuffle the random sequence in place i.e., it changes the position of items in a list.
    Shuffle method takes two parameters. Out of the two random is an optional parameter"""
from random import shuffle
 
 
# The sequential model allows you to create models layer-by-layer for most problems.
from keras.models import Sequential
 
# Long Short-Term Memory (LSTM) networks are an extension for recurrent neural networks(RNN), which basically extends their memory.
from keras.layers.recurrent import LSTM
 
""" Dense implements the operation: output = activation(dot(input, kernel) + bias)
    where Activation is the element-wise activation function passed as the activation argument.
    Dropout is a technique where randomly selected neurons are ignored during training"""
from keras.layers.core import Dense, Activation, Dropout
 
""" A callback is a set of functions to be applied at given stages of the training procedure.
You can use callbacks to get a view on internal states and statistics of the model during training"""
 #CSVLogger = Callback that streams epoch results to a csv file.
 #TensorBoard = TensorBoard basic visualizations. TensorBoard is a visualization tool provided with TensorFlow.
 #EarlyStopping=Stop training when a monitored quantity has stopped improving.
from keras.callbacks import CSVLogger, TensorBoard, EarlyStopping
 
 
""" We can use Sequential Keras models (single-input only) as part of your Scikit-Learn workflow via the wrappers found at keras.wrappers.scikit_learn.py.
There are two wrappers available. They are
KerasClassifier, which implements the Scikit-Learn classifier interface,
KerasRegressor, which implements the Scikit-Learn regressor interface"""
from keras.wrappers.scikit_learn import KerasClassifier
 
 """ Exhaustive search over specified parameter values for an estimator.
GridSearchCV implements a “fit” and a “score” method.
It also implements “predict”, “predict_proba”, “decision_function”, “transform” and “inverse_transform”
if they are implemented in the estimator used.
The parameters of the estimator used to apply these methods are optimized by cross-validated grid-search over a parameter grid """
from sklearn.model_selection import GridSearchCV
 
 
import time # To handle time-related tasks. time() function returns the number of seconds passed since epoch.
 
import tensorflow as tf #TensorFlow is an open source library for fast numerical computing
 
import random as rn # random module will Generate a random value from the sequence
 
""" you basically set environment variables in the notebook using os.environ.
It's good to do the following before initializing Keras to limit Keras backend TensorFlow to use first GPU.
If the machine on which you train on has a GPU on 0, make sure to use 0 instead of 1 """
import os
os.environ['PYTHONHASHSEED'] = '0'
 
 
np.random.seed(42)
 # The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
 
rn.seed(12345)
# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
 
# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
 
 
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
 
from keras import backend as K
 
# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
 
 
tf.set_random_seed(1234)
 
 """ Using tensorflow we can create a session, define constants and perform computation with those constants using the session """  
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
 
 
def get_filepaths(mainfolder):
    """ Searches a folder for all unique files and compile a dictionary of their paths.
 Parameters
 --------------
 mainfolder: the filepath for the folder containing the data
 Returns
 --------------
 training_filepaths: file paths to be used for training
 testing_filepaths:  file paths to be used for testing
 """
    training_filepaths = {}
    testing_filepaths  = {}
    folders = os.listdir(mainfolder)
    for folder in folders:
        fpath = mainfolder + "/" + folder
        if os.path.isdir(fpath) and "MODEL" not in folder:
            filenames = os.listdir(fpath)
            for filename in filenames[:int(round(0.8*len(filenames)))]:
                fullpath = fpath + "/" + filename
                training_filepaths[fullpath] = folder
            for filename1 in filenames[int(round(0.8*len(filenames))):]:
                fullpath1 = fpath + "/" + filename1
                testing_filepaths[fullpath1] = folder
    return training_filepaths, testing_filepaths
 
def get_labels(mainfolder):
    """ Creates a dictionary of labels for each unique type of motion """
    labels = {}
    label = 0
    for folder in os.listdir(mainfolder):
        fpath = mainfolder + "/" + folder
        if os.path.isdir(fpath) and "MODEL" not in folder:
            labels[folder] = label
            label += 1
    return labels
 
def get_data(fp, labels, folders, norm, std, center):
    """ Creates a dataframe for the data in the filepath
    and creates a one-hot encoding of the file's label """
    data = pd.read_csv(filepath_or_buffer=fp, sep=' ', names = ["X", "Y", "Z"])
    if norm and not std:
        normed_data = norm_data(data)
    elif std and not norm:
        stdized_data = std_data(data)
    elif center and not norm and not std:
        cent_data = subtract_mean(data)
 
    one_hot = np.zeros(14)
    file_dir = folders[fp]
    label = labels[file_dir]
    one_hot[label] = 1
    return normed_data, one_hot, label
 
# Normalizes the data by removing the mean
 
def subtract_mean(input_data):
    # Subtract the mean along each column
    centered_data = input_data - input_data.mean()
    return centered_data
 
 
def norm_data(data):
    """ Normalizes the data.
    For normalizing each entry, y = (x - min)/(max - min)"""
    c_data = subtract_mean(data)
    mms = MinMaxScaler() # Transforms features by scaling each feature to a given range e.g. between zero and one
    mms.fit(c_data)
    n_data = mms.transform(c_data)
    return n_data
 
def standardize(data):
    c_data = subtract_mean(data)
    std_data = c_data/ pd.std(c_data)
    return std_data
 
def vectorize(normed):
    """ Uses a sliding window to create a list of (randomly-ordered) 300-timestep
        sublists for each feature"""
    sequences = [normed[i:i+300] for i in range(len(normed)-300)]
    shuffle(sequences)
    sequences = np.array(sequences)
    return sequences
 
def build_inputs(files_list, accel_labels, file_label_dict, norm_bool, std_bool, center_bool):
    X_seq    = []
    y_seq    = []
    labels = []
    for path in files_list:
        normed_data, target, target_label = get_data(path, accel_labels, file_label_dict, norm_bool, std_bool, center_bool)
        input_list = vectorize(normed_data)
        for inputs in range(len(input_list)):
            X_seq.append(input_list[inputs])
            y_seq.append(list(target))
            labels.append(target_label)
    X_ = np.array(X_seq)
    y_ = np.array(y_seq)
    return X_, y_, labels
 
# Builds the LSTM model
# 6 layers
def build_model():
    # baseline
    model = Sequential()
    #layer 1 - LTSM 128, activation is tanh  https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
    #create model
    model.add(LSTM(128, activation='tanh', recurrent_activation='hard_sigmoid',\
                    use_bias=True, kernel_initializer='glorot_uniform',\
                    recurrent_initializer='orthogonal',\
                    unit_forget_bias=True, kernel_regularizer=None,\
                    recurrent_regularizer=None,\
                    bias_regularizer=None, activity_regularizer=None,\
                    kernel_constraint=None, recurrent_constraint=None,\
                    bias_constraint=None, dropout=0.0, recurrent_dropout=0.0,\
                    implementation=1, return_sequences=True, return_state=False,\
                    go_backwards=False, stateful=False, unroll=False,\
                    input_shape=(300, 3)))
    #layer 2 - 50% dropout https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
    model.add(Dropout(0.5))
    #layer 3 - same as layer 1
    model.add(LSTM(128, activation='tanh', recurrent_activation='hard_sigmoid',\
                    use_bias=True, kernel_initializer='glorot_uniform',\
                    recurrent_initializer='orthogonal',\
                    unit_forget_bias=True, kernel_regularizer=None,\
                    recurrent_regularizer=None,\
                    bias_regularizer=None, activity_regularizer=None,\
                    kernel_constraint=None, recurrent_constraint=None, \
                    bias_constraint=None, dropout=0.0, recurrent_dropout=0.0,\
                    implementation=1, return_sequences=False, return_state=False,\
                    go_backwards=False, stateful=False, unroll=False,
                    input_shape=(300, 3)))
    #layer 4 - same as layer 2
    model.add(Dropout(0.5)) # Dropout is a technique where randomly selected neurons are ignored during training
    #layer 5 - for 2d data with tensor  https://keras.io/getting-started/sequential-model-guide/
    model.add(Dense(14)) #Dense implements the operation: output = activation(dot(input, kernel) + bias)
    #layer 6   softmax layer https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d
    model.add(Activation('softmax')) # where Activation is the element-wise activation function passed as the activation argument.
 
    start = time.time()
    # compile model
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    print("Compilation time: ", time.time(), '-', start)
 
    return model
 
def compute_accuracy(predictions, y_labels):
    predicted_labels = []
    for prediction in predictions:
        prediction_list = list(prediction)
        predicted_labels.append(prediction_list.index(max(prediction_list)))
    correct = 0
    for label in range(len(predicted_labels)):
        print("Predicted label: {}; Actual label: {}".format(predicted_labels[label], y_labels[label]))
        if predicted_labels[label] == y_labels[label]:
            correct += 1
    accuracy = 100 * (correct / len(predicted_labels))
    print("Predicted {} out of {} correctly for an Accuracy of {}%".format(corr# Raw sequence labeling with KnnDtwClassifier and KNN=1
clf1 = KnnDtwClassifier(1)    
clf1.fit(train_data_raw, train_labels)
 
for index, t in enumerate(test_data_raw):
    print("KnnDtwClassifier prediction for " +
          str(test_labels[index]) + " = " + str(clf1.predict(t)))ect, len(predicted_labels), accuracy))
    return
 
if __name__ == '__main__':
 
    if os.path.isdir("/Users/Sharoze"):
        mainpath = "/Users/Sharoze/ADL_Dataset/HMP_Dataset"
    else:
        mainpath = "~/ADL_Dataset"
 
    activity_labels                  = get_labels(mainpath)
    training_dict, testing_dict      = get_filepaths(mainpath)
    training_files                   = list(training_dict.keys())
    testing_files                    = list(testing_dict.keys())
 
    # build training inputs and labels
    X_train, y_train, train_labels = build_inputs(
        training_files,
        activity_labels,
        training_dict,
        True, False, False)
       
    # build testing inputs and labels
    X_test, y_test, test_labels    = build_inputs(
        training_files,
        activity_labels,
        training_dict,
        True, False, False)
 
    # build and run model
    epochs = 5 #200 originally set to 200 but taking too long to compile. Reduced to 5 but an increased amount will be appropriate 
    for test in range(epochs):
        model = build_model()
        #CSVLogger = Callback that streams epoch results to a csv file.
        csv_logger = CSVLogger('training.log', append=True)
        # launch TensorBoard via tensorboard --logdir=/full_path_to_your_logs
        tb_logs = TensorBoard(log_dir='./logs', histogram_freq=10,
        batch_size=32, write_graph=True, write_grads=True, write_images=True,
        embeddings_freq=25, embeddings_layer_names=None, embeddings_metadata=None)  
        #TensorBoard = TensorBoard basic visualizations. TensorBoard is a visualization tool provided with TensorFlow.
   
 
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=10,
                                        verbose=0, mode='auto')
        #EarlyStopping=Stop training when a monitored quantity has stopped improving.
        model.fit(X_train, y_train, epochs=epochs,
            validation_split=0.2, callbacks=[csv_logger, early_stop]) #, tb_logs])
 
       
        #using one-hot values to input numerical data as the model cant input labeled data  https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/
        pred = model.predict(X_test)
        print("Predicted one-hot values: {} \n Actual one-hot values: {}".format(pred, y_test))
        print("Prediction shape: {} \n Actual shape: {}".format(pred.shape, y_test.shape))
 
        compute_accuracy(pred, test_labels)