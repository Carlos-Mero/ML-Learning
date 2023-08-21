# Attention Mechanisms and Transformers

Almost all the dominating NLPs nowadays are using Transformers at its core.
Meanwhile, the vision Transformers are emerged as a default model for diverse
vision tasks too.

The fundamental idea behind the Transformers model is the attention mechanism,
an innovation that was originally envisioned as an enhancement for encoder-decoder
RNNs applied to sequence-to-sequence applications.

## Queries, Keys, and Values

Compare it to dataset, a model could give us multiple probable outputs due to
a single query, just as multiple values corresponding to the same key.

Rather than choosing one of them as the final result, we can alternativly choose
to calculate an average of all of them.
This is the original pasion behind the attention mechanism, i.e.:

$$\text{Attention}(q, \mathcal D)=\sum_{i=1}^m\alpha(q,k_i)v_i$$

Where $q$ is the query fed to the model, $v_i$ is the corresponding output vector,
and $\alpha(q,k_i)$ are scalar attention weights.

The operation itself is typically refered to as **attention pooling**.
The name attention derives from the fact that the operation pays particular
attention to the terms for which the weight $\alpha$ is significant (i.e., large).

We would expect a number of special cases:

* The weights $\alpha(q,k_i)$ are nonnegative, so that the output of the attention
    mechanism is contained in the convex cone spanned by the values $v_i$.
* The weights sum up to $1$, so that the weights form a convex combination.

These two conditions can be obtained by applying the softmax operation:

$$\alpha(q,k_i)=\frac{\exp(\alpha(q,k_i))}{\sum_j\exp(\alpha(q,k_j))}$$

Such is the primary components of the attention mechanism.

## Attention Pooling by Similarity

There're several methods to estimate the similarity kernel $\alpha(q,k)$
relating queries $q$ to keys $k$. Some common kernels are

$$\alpha(q,k)=\exp\left(-\frac12\|q-k\|^2\right)\quad\text{Gaussian}$$

$$\alpha(q,k)=1\quad\text{if }\|q-k\|\leq1\quad\text{Boxcar}$$

$$\alpha(q,k)=\max(0,1-\|q-k\|)\quad\text{Epanechikov}$$
