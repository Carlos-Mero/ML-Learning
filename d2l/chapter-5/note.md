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
