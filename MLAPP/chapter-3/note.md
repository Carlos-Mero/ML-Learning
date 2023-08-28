# Generative models for discrete data

## Introduction

A typical Bayes model should look like this:

$$p(y=c|x,\theta)\propto p(x|y=c,\theta)p(y=c|\theta)$$

The key to using such models is specifying a suitable form for the class-conditional
density $p(y=c|x,\theta)$.

## Bayesian concept learning

Concept learning is like learning the meaning of a word, which in turn
is equivalent to binary classification.
If we difine $f(x)=1$ if $x$ is an example of the concept $C$, and
$f(x)=0$ otherwise, by alowing for uncertainty about the defination of $f$,
or equivalently the elements of $C$, we can emulate **fuzzy set theory**.

The key intuition of choosing from a hypothesis space is to avoid **suspicious
coincidence**. To formalize this, we may introduce the **strong sampling assumption**,
assuming that the examples are sampled uniformly at random from the **extension** of
a concept.

Then the probability of independently sampling $N$ items (with replacement) from $h$
is given by

$$p(D|h)=\left[\frac1{\text{size}(h)}\right]^N=\left[\frac1{|h|}\right]^N$$

This crucial equation embodies what is called the **size principle**, which
means the model favors the simplest (smallest) hypothesis consisitent with the data.
This is more commonly known as **Occam's razor**.

**Prior** is the mechanism by which background knowledge can be brought to bear
on a problem. Without this, rapid learning is impossible.

The **posterior** is calculated by the Bayes' rule.

$$p(h|D)=\frac{p(\mathcal D|h)p(h)}{\sum_{h'\in\mathcal H}p(\mathcal D,h')}
=\frac{p(h)\mathbb I(\mathcal D\in h)/|h|^N}{\sum_{h'\in\mathcal H}p(h')\mathbb I
(\mathcal D\in h')/|h'|^N}$$

where $\mathbb I(\mathcal D\in h)$ is 1 iff all the data are in the extension of
the hypothesis $h$, and 0 otherwise.

In general, if we have enough data, the posterior will converge to

$$p(h|\mathcal D)\to\delta_{\hat h^{MAP}}(h)$$

where $\hat h^{MAP}=\argmax_hp(h|\mathcal D)$ is the posterior mode, and $\delta$
is the **Dirac measure**.

This is call the MAP estimate.
And since the likelihood term depends exponentially on $N$, the MAP estimate
converges towards the **maximum likelihood estimate** as we get more and more data.

If the true hypothesis is in the hypothesis space, then the MAP/ML estimate will
converge upon this hypothesis. Thus we say that Bayesian inference are consisitent
estimators, and the hypothesis space is **identifiable in the limit**.

The posterior is our internal **belief state** about the world.
Specifically, the posterior predictive distribution in this context is given by

$$p(\bar x\in C|\mathcal D)=\sum_hp(y=1|\bar x,h)p(h|\mathcal D)$$

This is the weighed average of the predictions of each individual hypothesis and
is called **Bayes model averaging**.

**Conjugate Prior** has already been discussed by the previous notes.

The posterior mean is convex combination of the prior mean and the MLE,
which captures the notion that the posterior is a compromise between what we
previously believed and what the data is telling us.
Also, the posterior mode is a convex combination of the prior mode and the MLE.

There is an **add-one smoothing** technique which helps to prevent zero count problem
or **sparse data problem**.
