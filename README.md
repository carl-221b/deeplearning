# Convolutional Neural Network

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

### Training

I've choosen 10 epochs and 128 batch_size for the training since it seems like the accuracy reaches its peak pretty fast and is steady from there.

![](https://raw.githubusercontent.com/carl-221b/deeplearning/main/results/mnist_accuracy.png "accuracy")

![](https://raw.githubusercontent.com/carl-221b/deeplearning/main/results/mnist_loss.png "loss")

The final accuracy is 99.2%

### Predictions 
Here's some of the correct results with the highest probability.

![](https://raw.githubusercontent.com/carl-221b/deeplearning/main/results/mnist_good.png "correct")

The worst ones.

![](https://raw.githubusercontent.com/carl-221b/deeplearning/main/results/mnist_bad.png "incrrect")

## Cifar 10

### Dataset

The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.  
  
The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

Here's some example of the cifar classes and the images in the dataset in these classes.

![](https://raw.githubusercontent.com/carl-221b/deeplearning/main/results/cifar%20example.png "cifar example")

### Network
The full neural network witha ll the layers.
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 32, 32, 32)        896       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 32, 32, 32)        9248      
_________________________________________________________________
batch_normalization (BatchNo (None, 32, 32, 32)        128       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 16, 16, 32)        0         
_________________________________________________________________
dropout (Dropout)            (None, 16, 16, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 16, 16, 64)        18496     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 16, 16, 64)        36928     
_________________________________________________________________
batch_normalization_1 (Batch (None, 16, 16, 64)        256       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 8, 8, 64)          0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 8, 8, 64)          0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 8, 8, 128)         73856     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 8, 8, 128)         147584    
_________________________________________________________________
batch_normalization_2 (Batch (None, 8, 8, 128)         512       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 4, 4, 128)         0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 4, 4, 128)         0         
_________________________________________________________________
flatten (Flatten)            (None, 2048)              0         
_________________________________________________________________
dense (Dense)                (None, 128)               262272    
_________________________________________________________________
dropout_3 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290      
=================================================================
```

### Training

I trained this model over 100 epochs with 64 batch_size.

![](https://raw.githubusercontent.com/carl-221b/deeplearning/main/results/accuracy_3_3.png)
![](https://raw.githubusercontent.com/carl-221b/deeplearning/main/results/loss_3_3.png)

I got a final accuracy of  86.87%.

### Predictions
Some random predictions i have collected
![](https://cdn.discordapp.com/attachments/756504949583511601/767539248479600730/wHv6M6tcchPgAAAABJRU5ErkJggg.png)


The worst predictions i have collected with this model.
![](https://raw.githubusercontent.com/carl-221b/deeplearning/main/results/results.png)
