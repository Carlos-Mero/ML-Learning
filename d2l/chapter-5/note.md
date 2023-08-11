# Convolutional Neural Networks

Image data is represented as a two-dimensional grid of pixels, be it
monochromatic or in color.
In the previous chapter we've tried to flatten the images and send it to
a fully connected layer, and expect it to work.
The fact is that it actually works, with a performance that is not that
satisfying.

However, if we could recall our prior knowledge that the pixels in an image
should be spatially relevant to each other.
This leads to the concept of **Convolutional Neural Networks**(CNNs), a
powerful family of neural networks that are designed for precisely this
purpose. CNN will provide significant performance improvements in image
collection or other works.

## From Fully Connected Layers to Convolutions

We have two important priors here when doing with the image.
They are

* translation invariance
* locality

With these principles, we can significantly simplify our fully connected
layer, cut down the parameters we need by orders of magnitude.

And then, the final result we get is just the convolution.

## Convolutions for Images

A convolutional layer actually describes cross-correlations.
We're using the discret version of convolution to calculate such correlations.
The training process of a convolutional layer is to learn the convolution
kernel.

The output of the convolutional layer is also called a feature map,
as it can be regarded as the learned representations in the spatial
dimensions.
And for some element $x$ in some layer, its receptive field refers to all
the elements from the previous layer that may affect the calculation of the
element during the forward propagation.

## Padding and Stride

Padding and stride convolutions are two kinds of techniques that offer more
control over the size of the output.
These techniques are quite easy to understand.

The first one is implemented by adding zeros around the edge of the
2-dimensional data, thus our convolution kernels would be able to be
applied to a wider range of points.

Most of the time we tend to expand the input image so that we can get an
output of the same shape of the original input tensor.
This can be obtained by simply using `nn.LazyConv2d` method.

Stride is another technique that gives the opposite effect.
As its name conveys, "stride" means to move the convolution kernel by
more than one steps after a cross-correlation process, which will result in
much fewer outputs.
This can be quite useful if we want to downsample or aiming for computational
efficiency.

## Multiple Input and Multiple Output Channels

In the case that we may need to handle multiple inputs, we can simply
transfer to multi-dimensional convolution kernels instead, and repeat
the same process as before.

## Pooling

Pooling layers serve the dual purpose of mitigating the sensitivity of
convolutional layers to location and of spatially downsampling representations.

Polling operators consist of a fixed-shape window that is slid over all
regions in the input according to its stride, just like that in a convolutional
layer. However, a polling layer contains no parameters, most of the time
they simply calculate the maximum or the average value of the elements in the
pooling window, as a deterministic operator.

## Convolutional Neural Network LeNet

LeNet is the one among the first published CNNs that captures wide attention
for its performance on computer vision tasks.

At a high level, LeNet(LeNet-5) consists of two parts and several layers.
The first part is a convolutional encoder consisting of two convolutional
layers; and the last part is a dense block consisting of three connected
layers.
The original LeNet uses sigmoid function as its activation function, along
with a subsequent average pooling operation.
In fact ReLUs and max-pooling work better, but these discoveries had not yet
been made at the time.

Again, we'll use LeNet for image classification on the FashionMNIST dataset.
