# Probability

## Probability Theory Basis

The fundamentals of the probability theorem is repeated over and over again.
I don't want to mark down too much here, but write down some terms to memory.

* random variables <=> rv
* probability mass function <=> pmf
* conditionally independent <=> CI
* cumulative distribution function <=> cdf
* probability density function <=> pdf
* independent and identically distributed <=> iid

> Some common distributions

* **Student $t$ distribution**

One shortage of the normal distributions is that it is quite sensitive to outliers
since the pdf decays very quickly,
to solve such problem a more robust distribution is modified
to replace the normal distribution, called **Student $t$ distribution**:

$$\Tau(x|\mu,\sigma^2,\nu)\propto
\left(1+\frac1\nu\left(\frac{x-\mu}{\sigma}\right)^2\right)^{-(\frac{\nu+1}2)}$$

$$\text{mean}=\mu,\text{mode}=\mu,\text{var}=\frac{\nu\sigma^2}{(\nu-2)}$$

where $\mu$ is the mean, $\sigma^2>0$ is the scale parameter, and $\nu>0$
is called the **degree of freedom**.

The variance is only defined if $\nu>2$, and the mean is only defined if $\nu>1$.
In fact, if $\nu=1$, this distribution is known as the **Cauchy Distribution**.
To ensure finite variance, we require $\nu>2$, and it is common to use $\nu=4$
which gives good performance in a range of problems.

* **The Laplace distribution**

Another distribution with heavy tails is the **Laplace distribution**, also
known as **double sided exponential distribution**, the pdf is:

$$\text{Lap}(x|\mu,b):=\frac1{2b}\exp\left(-\frac{|x-\mu|}b\right)$$

$$\text{mean}=\mu,\text{mode}=\mu,\text{var}=2b^2$$

where $\mu$ is a location parameter and $b>0$ is a scale parameter.

* **The gamma distribution**

The gamma distribution is a flexible distribution for positive real valued rv's.
It is defined in terms of two parameters, called the shape $a>0$ and the rate
$b>0$:

$$\text{Ga}(T|\text{shape}=a,\text{rate}=b):=\frac{b^a}{\Gamma(a)}T^{a-1}\exp(-Tb)$$

where $\Gamma(x)$ is defined by:

$$\Gamma(x):=\int_0^\infty u^{x-1},\exp(-u)\mathrm du$$

The gamma distribution has the following properties:

$$\text{mean}=\frac ab,\text{mode}=\frac{a-1}b,\text{var}=\frac a{b^2}$$

There are several distributions that are special cases of Gamma, they are

* Exponential distribution, $\text{Expon}(x|\lambda)=\text{Ga}(x|1,\gamma)$.
* Erlang distribution $\text{Erlang}(x|\lambda)=\text{Ga}(x|1,\gamma)$.
* Chi-squared distribution, $\chi^2(x|\nu)=\text{Ga}(x|\frac\nu2,\frac12)$.

Another useful result is that if $X\sim\text{Ga}(a,b)$, then one can show that
$\frac1X\sim\text{IG}(a,b)$, where $\text{IG}$ is the **inverse gamma**
distribution defined by:

$$\text{IG}(x|\text{shape}=a,\text{scale}=b):=
\frac{b^a}{\Gamma(a)}x^{-(a+1)}\exp(-\frac bx)$$

$$\text{mean}=\frac b{a-1},\text{mode}=\frac b{a+1},
\text{var}=\frac{b^2}{(a-1)^2(a-2)}$$

* **The beta distribution**

The beta distribution has support over the interval $[0,1]$, and is defined as follows:

$$\text{Beta}(x|a,b)=\frac1{B(a,b)}x^{a-1}(1-x)^{b-1}$$

where $B(p,q)$ is the beta function,

$$B(a,b):=\frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}$$

The beta distribution has the following properties:

$$\text{mean}=\frac a{a+b},\text{mode}=\frac{a-1}{a+b-2},
\text{var}=\frac{ab}{(a+b^2)(a+b+1)}$$

* **Pareto distribution**

The pareto distribution is used to model the distribution of quantities that
exhibit long tails, also called heavy tails.
The pdf of pareto distribution is defined by:

$$\text{Pareto}(x|k,m)=km^kx^{-(k+1)}\chi_{x\geq m}$$

This density asserts that $x$ must be greater than some constant $m$, but
not too much greater.
And below are the properties of the Pareto distribution.

$$\text{mean}=\frac{km}{k-1},\text{mode}=m,\text{var}=\frac{m^2k}{(k-1)^2(k-2)}$$

And the variance only exists when $k>2$.

And we'll skip about more reciption on the probability theory.

## Information Theory Basis

Infotmation theory is concerned with representing data in a compact fasion.

* **Entropy**

The entropy of a random variable $x$ with distribution $p$ is denoted by
$\mathbb H(x)$ or sometimes $\mathbb H(p)$, is a measure of its uncentainly.
For a discrete rv, the entropy is defined by:

$$\mathbb H(x):=-\sum_kp(x=k)\log_2p(x=k)$$

Usually we use log base 2, in which case the units are called **bits**
(short for binary bits). And if we use log base $e$, the units are called
**nats**. The sum will get replaced by an integral for pdfs.
For the special case of binary rvs, $x\in\{0,1\}$, the entropy is called binary
entropy function.

* **KL divergence**

**Kullback-Leibler divergence (KL divergence)** or **relative entropy**
is one way to measure the dissimilarity of two probability distributions,
$p$ and $q$. It is defined as follows:

$$\mathbb{KL}(p\|q):=\sum_kp_k\log\frac{p_k}{q_k}$$

where the sum gets replaced by an integral for pdfs.
We can also rewrite it as:

$$\mathbb{KL}(p\|q)=\sum_kp_k\log p_k-\sum_kp_k\log q_k=-\mathbb H(p)+\mathbb H(p,q)$$

where $\mathbb H(p,q)$ is called the **cross entropy**.

Note that the KL divergence can not be a distance since it is asymmetric.
There's one way to fix this though.

Cross entropy is just the average number of bits needed to encode data coming
from a source with distribution $p$ when we're using model $q$ to define our
codebook. The "regular" entropy $\mathbb H(p)= \mathbb H(p,p)$ is just the
smallest requirement, so the KL divergence is the difference between these,
reflects the average number of extra bits needed to encode the data.

This is the **Theorem on Information Inequality**:

$$\mathbb{KL}(p\|q)\geq0\quad\text{with equality iff }p=q$$

This can be obtained by using Jensen's inequality.

One important consequence of this result is that the discrete distributions
with the maximum entropy is the uniform distribution.

* **Mutual information**

**Mutual information**, or **MI** in short, is a general approch to determine
how similar the joint distribution $p(x,y)$ is to the factored distribution
$p(x)p(y)$, for two rvs $x$ and $y$.
It has much wider usage compared to correlation coefficient.
The MI is defined by:

$$\mathbb I(x;y):=\mathbb{KL}(p(x,y)|p(x)p(y))=
\sum_x\sum_yp(x,y)\log\frac{p(x,y)}{p(x)p(y)}$$

Due to information inequality, the MI is zero iff the variables are independent.
To gain insight into the meaning of MI, we can re-express it in terms of joint
and conditional entropies as below:

$$\mathbb I(x;y)=\mathbb H(x)-\mathbb H(x|y)=\mathbb H(y)-\mathbb H(y|x)$$

where $\mathbb H(y|x)$ is called the **conditional entropy**, defined as
$\mathbb H(y|x)=\sum_kp(x)\mathbb H(y|x=k)$

A quantity which is closely related to MI is the **pointwise mutual information**
or **PMI**. For two events $x$ and $y$, this is defined as

$$\text{PMI}(x,y):=
\log\frac{p(x,y)}{p(x)p(y)}=\log\frac{p(x|y)}{p(x)}=\log\frac{p(y|x)}{p(y)}$$

The MI of the two variables is just the expected value of the PMI.
