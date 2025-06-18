import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential
import matplotlib.pyplot as py


recycFolder_train = 'Tensorflow/pixyTrash/data/Recyclables'
compFolder_train =  'Tensorflow/pixyTrash/data/Compost'
recycFolder_test =  'Tensorflow/pixyTrash/data/Recyclables_test'
compFolder_test =  'Tensorflow/pixyTrash/data/Compost_test'


for fname in os.listdir(recycFolder_train):
    full_path = os.path.join(recycFolder_train, fname)

    