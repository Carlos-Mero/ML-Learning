# Gaussian Models

**Multivariate Gaussian** or **multivariate normal** (**MVN**) si the most widely
used joint probability density function for continuous variables.

The Mahalanobis distance between a data vector $x$ and the mean vector $\mu$ is
defined by $(x-\mu)^T\Sigma^{-1}(x-\mu)$.

The contours of equal probability density of a Gaussian lie along ellipses.

**Theorem:** MLE for a Gaussian

The MLE of a Gaussian is just the empirical mean and empirical covariance, that is

$$\hat\mu=\frac1N\sum_ix_i=\bar x$$

$$\hat{\sigma^2}=\frac1N\sum_i(x_i-\bar x)^2$$

And here are some results from matrix algebra

$$\frac{\partial(b^Ta)}{\partial a}=b$$

$$\frac{\partial(a^TAa)}{\partial a}=(A+A^T)a$$

$$\frac\partial{\partial A}\text{tr}(BA)=B^T$$

$$\frac\partial{\partial A}\log|A|=A^{-T}$$

$$\text{tr}(ABC)=\text{tr}(CAB)=\text{tr}(BCA)$$

The last equation is called the **cyclic permutation property** of the trace operator,
using which we can derive the widely used **trace trick** as follows:

$$x^TAx=\text{tr}(x^TAx)=\text{tr}(xx^TA)=\text{tr}(Axx^T)$$

**Theorem:** Gaussian has the maximum entropy subject to having a specified mean
and covariance.

This can be obtained by applying the inequality of information.

Note that the MLE of a Gaussian can badly overfit in high dimensions.
So we have to use a series of methods to prevent such condition.

**Theorem:** Marginals and conditionals of an MVN.

Suppose $x=(x_1,x_2)$ is jointly Gaussian with parameters:

$$\mu=\left(\begin{array}{c}\mu_1\\\mu_2\end{array}\right),
\Sigma=\begin{pmatrix}\Sigma_{11} & \Sigma_{12}\\
\Sigma_{21} & \Sigma_{22}\end{pmatrix},
\Lambda=\Sigma^{-1}=\begin{pmatrix}\Lambda_{11} & \Lambda_{12}\\
\Lambda_{21} & \Lambda_{22}\end{pmatrix}$$

Then the marginals are given by

$$
p(x_1) = \mathcal N(x_1|\mu_1,\Sigma_{11})\\
p(x_2) = \mathcal N(x_2|\mu_2,\Sigma_{22})
$$

and the posterior conditional is given by

$$
p(x_1|x_2) = \mathcal N(x_1|\mu_{1|2},\Sigma_{1|2})\\
\mu_{1|2}=\mu_1+\Sigma_{12}\Sigma_{22}^{-1}(x_2-\mu_2)\\
=\mu_1-\Lambda_{11}^{-1}\Lambda_{12}(x_2-\mu_2)\\
=\Sigma_{1|2}(\Lambda_{11}\mu_1-\Lambda_{12}(x_2-\mu_2))\\
\Sigma_{1|2}=\Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}=\Lambda_{11}^{-1}
$$

**Information form** of the MVN

Suppose $x\sim\mathcal N(\mu,\Sigma)$, this is called the **moment parameters** of
the distribution, however, it is sometimes useful to use the **canonical** or
**natural parameters**, defined as

$$\Lambda:=\Sigma^{-1},\xi:=\Sigma^{-1}\mu$$

$$\mu=\Lambda^{-1}\xi,\Sigma=\Lambda^{-1}$$

Using the cannonical parameters, we can write the MVN in **information form**:

$$
\mathcal N_c(x|\xi,\Lambda) = (2\pi)^{-D/2}|\Lambda|^{\frac12}\exp
\left(
-\frac12(x^T\Lambda x+\xi^T\Lambda^{-1}\xi-2x^T\xi)
\right)
$$

Note: given the assumption that

$$
M=\begin{pmatrix}
E & F \\
G & H
\end{pmatrix}
$$

And $E,H$ are invertible, then we can write

$$
M/H:=E-FH^{-1}G\\
M/E:=H-GE^{-1}F
$$

We have equation here

$$|M|=|M/H||H|$$
