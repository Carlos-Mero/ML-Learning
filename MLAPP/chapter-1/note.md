# Introduction

Machine learning is regarded as a set of methods that can automatically
detect patterns in data.
And this book adopts the view that the best way to solve such problems
is to use the tools of probability theory.

There're three main types of machine learning in total, the supervised
learning, the unsupervised learning, and reinforcement learning.
Reinforcement learning is somewhat less commonly used, and it is useful
for learning how to act or behave when given occasional reward or punisment
signals.

We will denote the probability distribution over possible labels, given
the input vector $\mathbf x$ and training set $\mathscr D$ as

$$p(y|\mathbf x,\mathscr D)$$

This implicitly assumes the model we choiced to fit the dataset as a
precondition. In some cases, i.e. if we want to choose between defferent
models, we will make this assumption explicit by writing

$$p(y|\mathbf x,\mathscr D,M)$$

where $M$ denotes the model.

In this book we'll focus on probabilistic models of the form
$p(\mathbf x)$ or $p(y|\mathbf x)$.

There're some basic concepts in machine learning:

* Parametric and non-parametric models.

    The parametric models are models with fixed number of data, and the latter
    are models whose number of paramaters can grow with the amount of training
    data. The non-parametric models are actually more flexible and expressive,
    but would require much more computation for larger datasets.
    The parametric models do quite better in such sense.

All models are wrong, but some models are useful.
