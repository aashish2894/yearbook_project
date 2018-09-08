from os import path
import util
from util import *
import numpy as np
from run import *
from matplotlib import pyplot as plt
from skimage.io import imshow
import pandas as pd

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten, Conv2D, MaxPool2D
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint

from keras.models import load_model

DATA_PATH = "/Users/aashishkumar/Documents/MS_3rd_Sem/Deep_Learning_Seminar/yearbook"

def createDataset(train,valid):
    list = util.listYearbook(train,valid)
    labels = np.array([float(y[1]) for y in list])
    labels = (labels-1905)/(2013-1905)
    images_array = []
    if train:
        YEARBOOK_LOAD_PATH = path.join(DATA_PATH, 'train')
    if valid:
        YEARBOOK_LOAD_PATH = path.join(DATA_PATH, 'valid')
    for image_path in list:
        image_full_path = path.join(YEARBOOK_LOAD_PATH, image_path[0])
        img = load(image_full_path)
        images_array.append(img)

    images_array_np = np.array(images_array)
    return images_array_np, labels


def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


def r_square_np(y_true,y_pred):
    SS_res = np.sum(np.square(y_true - y_pred))
    SS_tot = np.sum(np.square(y_true - np.mean(y_true)))
    epsilon = 1e-7
    return (1 - SS_res/(SS_tot + epsilon))

#Create Training Dataset
X_train, Y_train = createDataset(True,False)
X_train = X_train.astype('float32')
print("Train Dataset")
print(X_train.shape)
print(len(np.unique(Y_train)))
print(X_train.dtype)

#Create validation dataset
X_val, Y_val = createDataset(False,True)
X_val = X_val.astype('float32')
print("Validation Dataset")
print(X_val.shape)
print(len(np.unique(Y_val)))
print(X_val.dtype)


print("Training start........")
model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = (5,5),padding = 'Same',
	             activation ='relu', input_shape = (28,28,3)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same', activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64,activation='relu'))
model.add(Dense(1,))

epochs = 10
batch_size = 128
adam = optimizers.Adam(lr=0.001)
model.compile(loss='mean_squared_error', optimizer=adam)
#model.compile(loss='mean_squared_error', optimizer=adam, metrics=[r_square])
print(model.summary())

checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)
history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_val, Y_val), callbacks=[checkpointer], verbose = 2)

model.save('model_trained.h5')
#model.load_weights('weights.hdf5')

test_output = model.predict(X_val)
print(test_output)
print(test_output.shape)
test_output_label = test_output*(2013-1905) + 1905
test_output_label = np.squeeze(test_output_label)
test_output_label = np.round(test_output_label)
list = util.listYearbook(False,True)
true_labels = np.array([float(y[1]) for y in list])
print('R squared : ',r_square_np(true_labels,test_output_label))
test_output_label = pd.Series(test_output_label,name="Label")
true_labels = pd.Series(true_labels,name="True_Label")
submission = pd.concat([true_labels,test_output_label],axis = 1)

submission.to_csv("neural_network2.csv",index=False)

del model
model = load_model('model_trained.h5')

test_output = model.predict(X_val)
print(test_output)
print(test_output.shape)
test_output_label = test_output*(2013-1905) + 1905
test_output_label = np.squeeze(test_output_label)
test_output_label = np.round(test_output_label)
list = util.listYearbook(False,True)
true_labels = np.array([float(y[1]) for y in list])
print('R squared Saved Model: ',r_square_np(true_labels,test_output_label))
test_output_label = pd.Series(test_output_label,name="Label")
true_labels = pd.Series(true_labels,name="True_Label")
submission = pd.concat([true_labels,test_output_label],axis = 1)

submission.to_csv("neural_network_saved_model.csv",index=False)
