import numpy as np
import glob
import time
import scipy

import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D
import matplotlib.pyplot as plt
import glob
import datetime
import sklearn
import keras
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# DIRECTORY TO GET TEST AND TRAINING DATA FROM
dir = 'D:1250x8xall/'

train_reload = np.loadtxt(dir + 'train_data.txt').astype(np.float64)
test_reload = np.loadtxt(dir + 'test_data.txt').astype(np.float64)
#val_reload = np.loadtxt(dir + 'val_data.txt').astype(np.float64) 

# NUMBER OF POINTS 
no_points = 1250
num_feats = 6

# separates at 40 epochs


print("Loaded data files...")
# Shaping the training/testing data
# Shapes the data into an nx500x8 array
# Each sequence is 500 timesteps (10 ms of data collection)

train_data = np.reshape(train_reload,(-1,no_points,num_feats))
test_data = np.reshape(test_reload,(-1,no_points, num_feats))
#val_data = np.reshape(val_reload,(-1,no_points, num_feats))

train_label = np.loadtxt(dir + 'train_label.txt').astype(int)
test_label = np.loadtxt(dir + 'test_label.txt').astype(int)
#val_label = np.loadtxt(dir + 'val_label.txt').astype(int)

# train_data = scipy.fft.fft(train_data)
# test_data = scipy.fft.fft(test_data)
class EarlyStoppingByLossVal(keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', value=0.0001, verbose=0):
        super(keras.callbacks.Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            print("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True
#keeps track of the time spent on each epoch
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


model = Sequential()


filter_size = 100
kernel_size = 10

n_timesteps, n_features, output_size = train_data.shape[1], train_data.shape[2], train_label.shape[1]

model.add(tf.keras.Input(shape=(n_timesteps,n_features)))
model.add(tf.keras.layers.Conv1D(filter_size, kernel_size, activation='relu'))

model.add(tf.keras.layers.Conv1D(filter_size, kernel_size, activation='relu'))

model.add(tf.keras.layers.MaxPooling1D(3))

model.add(tf.keras.layers.Conv1D(filter_size, kernel_size, activation='relu')) 

model.add(tf.keras.layers.Conv1D(filter_size, kernel_size, activation='relu'))

model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dropout(.5))

model.add(tf.keras.layers.Dense(output_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer= tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
#model.compile(loss='binary_crossentropy', optimizer= tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
checkpoint_filepath = 'checkpointtest/'
#pred = model.predict(val_data)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

time_callback = TimeHistory()
early_stop = EarlyStoppingByLossVal()

# CHANGE EPOCH AND BATCH SIZE HERE
history = model.fit(train_data, train_label, validation_data=(test_data, test_label), epochs=100, batch_size=64, callbacks = [model_checkpoint_callback, time_callback, early_stop])
times = time_callback.times

print("timestamp:")
print(time_callback.times)
#np.savetxt("test/data2/timestampbin.txt", time_callback.times)
# Data cardinality is ambiguous:
#   x sizes: 8430
#   y sizes: 38580

model.summary()

_, test_accuracy = model.evaluate(test_data, test_label)
print(f"test accuracy: {test_accuracy}")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['loss', 'val'], loc='upper right')
plt.show()

# SAVE MODEL NAME / DIRECTORY
model.save('fft_model')


