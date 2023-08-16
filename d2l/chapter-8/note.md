# Modern Recurrent Neural Networks

## Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU)

While gradient clipping helps with exploding gradiens, handling vanishing gradients
appears to require a more elaborate solution.

One of the first and most successful techniques for addressing vanishing gradients
came in the form of the long short-term memory model.

LSTMs resemble standard recurrent neural networks but here each ordinry
recurrent node is replaced by a memory cell.
Each memory cell contains an internal state.

The term "long short-term memory" come from a simple intuition,
that we should have some long term memory that changes slowly during training,
encoding general knowledge about the data, and short term memory in the form
of ephemeral activations, which pass from each node to successive nodes.

### Gated Memory Cell

Each memory cell is equipped with an internal state and a number of multiplicative
gates that determine whether the internal state should interactive with input/output
or be changed in some way.

RNNs and particularly the LSTM architecture rapidly gained popularity during the
2010s. And since then a number of reserchers began to experiment with simplified
architectures in hopes of retaining the key idea of incorporating an internel state
and multiplicative gating mechanisms, but aim of speeding up computation.
The final result is the gated recurrent unit (GRU).

A GRU is mostly a simplified LSTM, where the three gates in that memory cell is
replaced by two.

Since LSTMs and GRUs are rarly used till this day, and are mostly replaced by
transformers in their application domains, we can just leave them a glance and
not spare too much attention on them.

## The Encoder-Decoder Architecture

The standard approach to handling inputs and outputs of varying lengths that are
unaligned is to design an encoder-decoder architecture.

The encoder firstly takes in inputs of various lengths, and outputs a fixed
length state. This state variable is then passed to the decoder to generate
the final outputs.

## Beam Search

Though costing quite little conputation, the greedy search method we used
before may not be the best one.
Since choosing the best suitable token at each step will tpically not lead us
the the golbal best solution, we'll have to seek for another selecting method
to gain a more prefered answer.

Though exhaustive search can guarantee the correctness of the result, it
would require an unreasonalbe computational cost.

Beam search is a compromise solution.
Beam search has several versions, the most straitforward one of which is
characterized by a single hyperparameter, the beam size $k$.
It captures the first $k$ candidate with the highest possibility.
Then we'll repeat the greedy search and obtain $k$ different sequences.
By picking the sequence with the highest probability, we can decide the final
output of the whole neural network.
