# Recurrent Neural Networks (RNNs)

RNNs are specially designed against the tasks that require dealing with sequential
data, both for inputs and outputs.
And in other domains, such as time series prediction, video analysis, and musical
information retrieval, a model must learn from inputs that sequences

The core structure of recurrent neural networks is recurrent connections,
which can be thought of as cycles in the network of nodes.

RNNs have recently ceded considerable market share to Transformers models.
But it is still worth paying attention to.

## Working with Sequences

Sequential models usually appears along with the probability.
Sometimes, especially when working with language, we wish to estimate the
joint probability of an entire sequence.

Most of the time we'll only consider the recent data when making predictions,
and won't look back so far.
In this case we might content ourselves to condition on some window length
$\tau$ and only use the recent ovservations.
The immediate benefit is that now the number of arguments is always the same,
at least for $t>\tau$. This allows us to train any models that requires fixed-
length vectors as inputs.

Generally, these sequence functions are called sequence models, and for natural
language data, they are called language models.
Because of the wide usage of language models, this name is also used to refer
to sequence models in many cases.

Whenever we can throw away the history beyond the precious $\tau$ steps without
any loss in predictive power, we say that the sequence satisfies a Markov condition,
i.e., that the future is conditionally independent of the past, given the
recent history.
When $\tau=k$, we say that the data is characterized by a k-th order Markov model.

## Converting Raw Text into Sequence Data

When working with text data represented as sequences of words, characters, or
word-piece, we'll need some preprocessing pipelines, typical steps are as follows.

* Load text as strings into memory
* Split the strings into tokens (e.g., words or characters)
* Build a vocabulary dictionary to associate each vocabulary element with a
    numerical index
* Convert the text into sequences of numerical indices.

## Language Models

The goal of language models is to estimate the joint probability of the
whole sequence where statistical tools can be applied.
Language models are incredibly useful, as we can see.

Laplace smoothing is a common strategy when dealing with word frequency.
The solution is to add a small constant to all counts, denote by
$n$ the total number of words in the training set and $m$ the number of
unique words, then

$$\hat P(x)=\frac{n(x)+\epsilon_1/m}{n+\epsilon_1}$$

Here $\epsilon_1$ is a hyperparameter.

It may bring us huge perplexity that considering how to measure the language
model quality.
Information theory comes handy here.
Since it is believed that a better language model should allow us to predict
the next token more accurately, we can measure it by the cross-entropy loss
averaged over all the $n$ tokans of a sequence:

$$\frac1n\sum_{t=1}^n-\log P(x_t|x_{t-1},\dots,x_1)$$

And due to historical reasons, scientists in natural language processing prefer
to use a quantity called perplexity. In a nutshell, it is the exponential of
the cross-entropy loss:

$$\exp\left(-\frac1n\sum_{t=1}^n\log P(x_t|x_{t-1},\dots,x_1)\right)$$

To iterate over (almost) all the tokens of the entire dataset for each epoch
and obtain all possible length-n subsequences, we can introduce randomness.
At the begining of the each epoch, discard the first $d$ tokens, where
$d\in[0,n)$ is uniformly sampled at random.
The rest of the sequences is then partitioned into $m=[(T-d)/n]$ subsequences.

For language modeling, the goal is to predict the next token based on what
tokens we have seen so far, hence the targets are the original sequence,
shifted by one token, i.e.:

![example from d2l](https://d2l.ai/_images/lang-model-data.svg)

## Recurrent Neural Networks

To control the scale of the model's parameters, it is preferable to use a
latent variable model:

$$P(x_t,x_{t-1},\dots,x_1)\approx P(x_t|h_{t-1})$$

where $h_{t-1}$ is a hidden state that stores the sequence informaltion up to
time step $t-1$.
In general, the hidden state at any time step $t$ could be computed based on
both the current input $x_t$ dan the previous hidden state $h_{t-1}$:

$$h_t=f(x_t,h_{t-1})$$

RNNs are neural networks with hidden states.
The key to RNNs is the recurrent layers.
They copy the hidden state from the previous steps and then pass it to the
following ones. Such process can be presented just like this:

![exapmle from d2l](https://d2l.ai/_images/rnn.svg)

If we choose to tokenize text into characters rather than words or something else,
and training our models based on such data type, this kind of language model is
called a character-level language model.

## Gradient Clipping

The influence from the first input would travel through a chain of $T$ layers
in the RNNs, and when taking backpropagate gradients through time, the result
could be in a chain of matrix-products with length $\mathscr O(T)$.
This would lead to great numerical instability, causing the gradients to
either explode or vanish depending on the properties of the weight matrices.

One inelegant but ubiquitous solution is to simply clip the gradients forcing
the resulting "clipped" gradients to take smaller values.

## Backpropagation Through Time

Backpropagation through time is a term specially refered to the backpropagation
method in RNNs.
By expanding the computational graph of an RNN this process is just a special
case discussed before, but with many more specific properties.

Complications arise because sequences can be rather long, and causing the
gradient computation to require many high level productions, e.g., 1000
matrix productions may be required to compute the gradient.

In practice we often uses truncated backpropagation through time, i.e.,
to truncate the sum of the gradient computation after $\tau$ steps,
and it works quite well.

We can also randomly choose the calculated steps $\tau$, and this is then
called the randomized truncation.

All the three kinds of calculating method can be discribed as below:

![computing gradients in RNNs](https://d2l.ai/_images/truncated-bptt.svg)

From top to bottom, randomized truncation, regular truncation, and full computation.
