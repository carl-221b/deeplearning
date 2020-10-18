from __future__ import print_function

#The two folloing lines allow to reduce tensorflow verbosity
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='1' # '0' for DEBUG=all [default], '1' to filter INFO msgs, '2' to filter WARNING msgs, '3' to filter all msgs

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing import image


import matplotlib.pyplot as plt
import numpy as np
import random
import os 
print('tensorflow:', tf.__version__)
print('keras:', tensorflow.keras.__version__)

##Uncomment the following two lines if you get CUDNN_STATUS_INTERNAL_ERROR initialization errors.
## (it happens on RTX 2060 on room 104/moneo or room 204/lautrec) 
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)



#load (first download if necessary) the MNIST dataset
# (the dataset is stored in your home direcoty in ~/.keras/datasets/mnist.npz
#  and will take  ~11MB)
# data is already split in train and test datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train : 60000 images of size 28x28, i.e., x_train.shape = (60000, 28, 28)
# y_train : 60000 labels (from 0 to 9)
# x_test  : 10000 images of size 28x28, i.e., x_test.shape = (10000, 28, 28)
# x_test  : 10000 labels
# all datasets are of type uint8
print('x_train.shape=', x_train.shape)
print('y_test.shape=', y_test.shape)

#To input our values in our network Conv2D layer, we need to reshape the datasets, i.e.,
# pass from (60000, 28, 28) to (60000, 28, 28, 1) where 1 is the number of channels of our images
img_rows, img_cols = x_train.shape[1], x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

#Convert to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Normalize inputs from [0; 255] to [0; 1]
x_train = x_train / 255
x_test = x_test / 255

print('x_train.shape=', x_train.shape)
print('x_test.shape=', x_test.shape)

num_classes = 10

#Convert class vectors to binary class matrices ("one hot encoding")
## Doc : https://keras.io/utils/#to_categorical
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)
# num_classes is computed automatically here
# but it is dangerous if y_test has not all the classes
# It would be better to pass num_classes=np.max(y_train)+1



#Let start our work: creating a convolutional neural network

mnist_shape= (28,28,1)
batch_size = 128
epochs = 15


def history(history):
  # Plot training & validation accuracy values
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper leftX')
  plt.show()

  # Plot training & validation loss values
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.show()


#####TO COMPLETE
def CNN_test():
    model = Sequential()
    model.add(Conv2D(7,kernel_size=(5,5),strides=(1,1),padding='same',input_shape=(32,32,3)))
    return model

def CNN_mnist():
    model = Sequential()
    model.add(Conv2D(32,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',input_shape = mnist_shape))
    model.add(Conv2D(64,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(num_classes,activation = 'softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    return model

if(os.path.isfile('mnist_model.h5')):
   mnist=tf.keras.models.load_model('mnist_model.h5')
   print ("Model found and loaded")

else:
   mnist=CNN_mnist()
   fit = mnist.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
   history(fit)
   score = mnist.evaluate(x_test, y_test, verbose=0)
   print('Test loss:', score[0])
   print('Test accuracy:', score[1])
   mnist.save("mnist_model.h5")

result_proba = mnist.predict(x_test)
correct_indices = np.nonzero((result_proba>0.5) == (y_test==1))[0]
incorrect_indices = np.nonzero((result_proba>0.5) != (y_test==1))[0]

## Pour plusieurs images
# charger les images

class_names = ['0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9']
#0 = mal classes
#1= bien classes
def plotResult (x):
    if(x==1):
        plt.figure("Bien classes",figsize=(10,10))
    else:
        plt.figure("Mal classes",figsize=(10,10))
    for i in range(25):
        if(x==1):
            index = correct_indices[random.randrange(correct_indices.shape[0])]
        else:
            index = incorrect_indices[random.randrange(incorrect_indices.shape[0])]
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_test[index], cmap=plt.cm.binary)
        proba = result_proba[index][np.argmax(y_test[index][:])] * 100
        proba = round (proba,3)
        plt.xlabel("VT: " + str(np.argmax(y_test[index][:]))+" P: " + str(np.argmax(result_proba[index][:])) + " (" + str(proba) + "%)" )

    plt.show()

plotResult(1)
plotResult(0)