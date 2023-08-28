# Exercise 3

**3.1.Proof:**

We can obtain the result by simply derive the likelihood of Bernoulli distribution.

$$p'(\mathcal D|\theta)=((1-\theta)N_1-\theta N_0)\theta^{N_1-1}(1-\theta)^{N_0-1}$$

$$p'(\mathcal D|\theta)=0\iff\theta=\frac{N_1}{N_0+N_1}$$

That's just the MLE of Bernoulli distribution.

**3.2.Proof:**

It is clear that the marginal likelihood is the ratio of the normalizing constants,
since the probability integrates to 1.
