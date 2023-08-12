# Modern CNNs

CNNs are well known in the computer vision and machine learning communities,
and during the current days, they actually dominate the field.

## Deep Convolutional Neural Networks (AlexNet)

Until 2012 the representation was calculated mostly mechanically.
After that, a group of reserchers have introduced the concept of representation
learning, along with the central idea of generating features from image data
automatically.

This first modern CNN is named AlexNet.
Interestingly in the lowest layers of the network, the model learned feature
extractors that resembled some traditional filters.

The AlexNet contains 5 convolutional layers and three full connected layers.
And between each layers there's a ReLU as its activation function.
It looks quite similar to the previous LeNet at the first glance.

AlexNet also uses dropout method to control the complexity of the model,
rather than weight decay which is used in LeNet.

## Networks using Blocks

Just as what we've done in chip design, we're now working with blocks rather
than layers when designing deep neural networks.
A decade later, this has even progressed to reserchers using entire trained
models to repurpose them for different, albeit related, tasks.
Such large pretrained models are typically called foundation models.

The idea of using blocks first emerged from the Visual Geometry Group (VGG)
at Oxford University.

A classical result shows that deeper networks significantly outperform shallower
counterparts while the scale of parameters are similar.

A VGG block contains

* a sequence of convolutions with $3\times3$ kernels with padding of 1.
* a $2\times2$ max-pooling layer with stride of 2.

## Network in Network

The full connected layer at the end of a neural network would consume tremendous
numbers of parameters, as well as a large amount of memory.
The network in network (NiN) blocks offers an alternative.
They were proposed based on a very simple insight:

* use $1\times1$ convolutions to add local nonlinearities across the channel.
* use global average pooling to integrate across all locations in the last
    representation layer.

## Multi-Branch Networks

GoogLeNet is the first neural network that combines the advantage of the
networks mentioned above.
The basic convolutional block in GoogLeNet is called an Inception Block,
which contains four branches in total.
It calculates convolutions of different sizes and combines them as an output.

## Batch Normalization

Training deep neural networks is difficult, and getting them to converge in a
reasonable amount of time can be tricky.
Batch normalization is a popular and effective technique that consistently
accelerates the convergence of deep networks.

Batch normalization can offer three benefits, just as follows

* preprocessing
* numerical stability
* regularization

Batch normalization is applied to individual layers, or optionally, to all
of them. In each training iteration, we first normalize the inputs by
subtracting their mean and dividing by their standard deviation, where
both are estimated based on the statistics of the current minibatch.
Next, we apply a scale coefficient and an offset to recover the lost degrees
of freedom.

Batch normalization implementations for fully connected layers and convolutional
layers are slightly different.
The main difference between the batch normalization layers for
full connected layer and convolutional layer is that the first one calculates
the mean and deviation according to the whole output tensor, while the
latter tend to focus on each output channels separately.

## Residual Networks and ResNeXt

A deeper and more complex neural network forms a larger function class, which
may lead to a better outcome when training on the same dataset.
This is guaranteed if the larger function class does contain the whole smaller
ones. This is the central idea of the residual network, that we should let
every additional layer to be more easily to contain the identity function as
one of its elements.

These considerations are rather profound but they led to a surprisingly simple
solution, a residual block.

This powerful tool is obtained by an addition operation called residual connection.
Residual connection means that after each newly added layer, we should add
the original input to the output, thus if we set all the parameters in this
layer to zero, we'll get the identity function.
And if the final output tensor has a different shape to the input, we can use
a $1\times1$ convolutional layer before adding to reshape the tensor.

## Densely Connected Networks (DenseNet)

ResNet significantly changed the view of how to parametrize the functions
in deep networks, and DenseNet is to some extent the logical extension of this.
In DenseNet, each layer in this model connects to all the preceding layers
and the concatenation operation to preserve and reuse features from earlier layers.

The key difference between ResNet and DenseNet is that in the latter case
outputs are concatenated, rather than added.
