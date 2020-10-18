# Convolutional Neural Network

This repository contains my experiments on Convolutional Neural Network using Tensorflow and Keras.

 ## Mnist database
The MNIST database of handwritten digits, available from this [page](http://yann.lecun.com/exdb/mnist/), has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.

### Neural network architecture 
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 28, 28, 32)        320       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 28, 28, 64)        18496     
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 64)        0         
_________________________________________________________________
dropout (Dropout)            (None, 14, 14, 64)        0         
_________________________________________________________________
flatten (Flatten)            (None, 12544)             0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 12544)             0         
_________________________________________________________________
dense (Dense)                (None, 128)               1605760   
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290      
=================================================================
```

### Correct prediction 

![](https://raw.githubusercontent.com/carl-221b/deeplearning/main/results/mnist_good.png "correct")

### Incorrect predictions

![](https://raw.githubusercontent.com/carl-221b/deeplearning/main/results/mnist_good.png "incrrect")
