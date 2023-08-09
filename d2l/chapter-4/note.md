# Builder's Guide

## Layers and Modules

Note that an MLP model we implemented will take in some inputs, while giving
out some outputs after a series of process.
This works just the same as that of a single layer of an MLP.

This introduces the concept of a neural network module, which can describe a
single layer, a component consisting of multiple layers, or the entire model
itself.

Such abstraction can help us to combine the existing models into larger
artifacts, often recursively.

The implementation of a custom module should look like this:

```python
class model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X):
        return X
```

And if we want to construct our own tiny `nn.Sequential`, a sample code could
look like this:

```python
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, X):
        for module in self.children():
            X = module(X)
        return X
```

And much more complicated process can be implemented in the `forward` method.
That's the basis of defining the modules in our neural network.

## Parameter Management

Parameter Management mainly focuses on two things here

* Accessing parameters for debugging, diagnostics, and visualizations.
* Sharing parameters across different model components.

We'll start with such simple model:

```python
import torch
from torch import nn
net = nn.Sequential(nn.LazyLinear(8),
                    nn.ReLU(),
                    nn.LazyLinear(1))

X = torch.rand(size=(2, 4))
net(X).shape
```

### Parameter Access

When a model is defined via the Sequential class, we can access any layer by
indexing into the model as though it were a list.
For example, if we want to inspect the parameters of the second fully connected
layers as follows:

```python
net[2].state_dict()
```

The output should contain both the identifiers and corresponding values of the
parameters. The parameters can be accessed by their identifiers too.

If we want to recursively through the entire tree of the model and access all
the parameters we have, here is the list we'll need:

```python
[(name, param) for name, param in net.named_parameters()]
```

Note the between multiple layers, the parameters between two directly connected
layers are tied. They're not just equal, they are actually the same.

## Parameter Initialization

Custom initialization of the parameters is a common requirement for training a
neural network.
The framework PyTorch already provides a series of commonly used protocols for
parameter initialization, and we can also create a custom one on our own.

By default, PyTorch initializes weight and bias matrices uniformly by drawing
from a range that is computed according to the input and output dimension.
Here is an example using the built-in initializers:

```python
def init_normal(module):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)

net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```

## Custom Layers

One factor behind deep learning's success is the availability of a wide range of
layers that can be composed in creative ways to design architectures suitable
for a wide variety of tasks.

As we mentioned before, a layer in a neural network is just a type of module,
thus a simple custom layer would look like

```python
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

This is a simple layer without parameters, and the most important thing for
creating such layer is to implement the forward propagation function.

As for the parameterized layers, we'll need to call the built-in function to
`nn.Parameter` define the parameters of the layer.
Take this as an example for a simple use

```python
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```

In this example, we need two arguments to initialize the parameters, which
separately denotes the dimension of the input and output tensors.
For some more complicated usage, the arguments could differ a lot.
The most significant thing to remember is that the method `nn.Parameter`
takes a tensor as an argument, and registers it to the framework.

## File I/O

File I/O is a fundamental requirement for a machine learning framework,
the most important one relevant function is `torch.save()`.
This function can take various objects from PyTorch as an argument,
and save them to a file. Examples could be:

```python
x = torch.arange(4)
torch.save(x, 'x-file')
```

The second argument of such function is just the path we'll save the data
to. More complicated usage can be found at file `io.py`.

And then, with the method `torch.load` to get the data we saved before.
