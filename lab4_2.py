from __future__ import print_function

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import RMSprop

import matplotlib.pyplot as plt
import numpy as np

import operator
import os.path


print('tensorflow:', tf.__version__)
print('keras:', tensorflow.keras.__version__)


##Uncomment the following two lines if you get CUDNN_STATUS_INTERNAL_ERROR initialization errors.
## (it happens on RTX 2060 on room 104/moneo or room 204/lautrec) 
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


#load (first download if necessary) the CIFAR10 dataset
# data is already split in train and test datasets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


#Let start our work: creating a convolutional neural network
img_rows, img_cols = x_train.shape[1], x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)

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
 
def history(history):
  # Plot training & validation accuracy values
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.savefig("accuracy_3_3.png")
  plt.show()
 
  # Plot training & validation loss values
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.savefig("loss_3_3.png")
  plt.show()

def create_model():
  model=Sequential()
  model.add(Conv2D(32,(3,3),(1,1),"same",activation='relu',input_shape=(32,32,3)))
  model.add(Conv2D(32,(3,3),(1,1),"same",activation='relu'))
  model.add(MaxPooling2D())
  model.add(Dropout(0.2))
  model.add(Conv2D(64,(3,3),(1,1),"same",activation='relu'))
  model.add(Conv2D(64,(3,3),(1,1),"same",activation='relu'))
  model.add(MaxPooling2D())
  model.add(Dropout(0.3))
  model.add(Conv2D(128,(3,3),(1,1),"same",activation='relu'))
  model.add(Conv2D(128,(3,3),(1,1),"same",activation='relu'))
  model.add(MaxPooling2D())
  model.add(Dropout(0.4))
  model.add(Flatten())
  model.add(Dense(128,activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(10,activation='softmax'))
  model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
  return model

if(os.path.isfile('model5.h5')):
   nn=tf.keras.models.load_model('model5.h5')
   print('loaded')
else:
  nn=create_model()

  batch_size = 64
  epochs = 100

  fit = nn.fit(x_train,y_train,validation_data = (x_test,y_test), batch_size=batch_size, epochs=epochs)
  nn.save("model5.h5")

  history(fit)

scores=nn.evaluate(x_test,y_test)
scores[1]=scores[1]*100 
print('Test Accuracy; %2f%%' %scores[1])
#scores[0] is loss, scores[1] is accuracy


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'Dog', 'frog', 'horse', 'ship', 'truck']

y_hat = nn.predict(x_test)

plt.figure(figsize=(14,14))

min = np.ones((25,2))

for i in range(y_hat.shape[0]):
  class_test = np.argmax(y_test[i][:])
  class_pred = np.argmax(y_hat[i][:])
  if(class_test != class_pred) :
    for j in range(25) :
      if min[j][0] > y_hat[i][class_pred] :
        min[j][0] = y_hat[i][class_pred]
        min[j][1] = i
        min = sorted(min, key=operator.itemgetter(0), reverse = True)
        break
min = sorted(min, key=operator.itemgetter(0))

for i in range(25):
    pic = (int)(min[i][1])
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[pic], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    class_test = np.argmax(y_test[pic][:])
    class_pred = np.argmax(y_hat[pic][:])
    percentage = str(int(y_hat[pic][class_pred]*100))
    plt.xlabel("VT : " + class_names[class_test] + ", Pred : " + class_names[class_pred] + " (" + percentage + "%)")
plt.savefig("results.png")
plt.show()