import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from timeit import default_timer as t

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# scaling image values between 0-1
X_train_scaled = X_train/255
X_test_scaled = X_test/255

# one hot encoding labels
y_train_encoded = keras.utils.to_categorical(y_train, num_classes = 10, dtype = 'float32')
y_test_encoded = keras.utils.to_categorical(y_test, num_classes = 10, dtype = 'float32')


def get_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(32, 32, 3)),
        keras.layers.Dense(3000, activation='relu'),
        keras.layers.Dense(1000, activation='relu'),
        keras.layers.Dense(10, activation='sigmoid')
    ])

    model.compile(optimizer='SGD',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

model = get_model()

print(model.summary())

print()
print()
print('*****************************************************************')
print()
print()

print('1 Epoch')
print()
# CPU
with tf.device('/CPU:0'):
    print('CPU')
    model_cpu = get_model()
    t1 = t()
    model_cpu.fit(X_train_scaled, y_train_encoded, epochs=1)
    t2 = t()
    T = t2-t1
    print('Time of training: {:.2f} s'.format(T))

print()
# GPU
if tf.test.is_built_with_cuda():
    with tf.device('/GPU:0'):
        print('GPU')
        model_gpu = get_model()
        t1 = t()
        model_gpu.fit(X_train_scaled, y_train_encoded, epochs=1)
        t2 = t()
        T = t2 - t1
        print('Time of training: {:.2f} s'.format(T))

print()
print()
print('*****************************************************************')
print()
print()

print('10 Epochs')
print()
# CPU
with tf.device('/CPU:0'):
    print('CPU')
    model_cpu = get_model()
    t1 = t()
    model_cpu.fit(X_train_scaled, y_train_encoded, epochs=10, verbose=0)
    t2 = t()
    T = t2-t1
    print('Time of training: {:.2f} s'.format(T))

print()

# GPU
if tf.test.is_built_with_cuda():
    with tf.device('/GPU:0'):
        print('GPU')
        model_gpu = get_model()
        t1 = t()
        model_gpu.fit(X_train_scaled, y_train_encoded, epochs=10, verbose=0)
        t2 = t()
        T = t2 - t1
        print('Time of training: {:.2f} s'.format(T))