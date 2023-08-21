# Exercise

**Exercise 2.1:**

* a. $2/3$
* b. $1/2$

**Exercise 2.2:**

* a. The probability can not be transfered between two events implicitly.
  No principles guarantees the correctness of such process.
* b. In fact, just in his own statement, this evidence has increased the probability
  that the one commited the crime by 10 times, that's a huge relevance between the
  two events.

**Exercise 2.3:**

This can be obtained just by mixing the defination with some calculation:

$$\text{var}[X+Y]=E(X+Y)^2-E^2(X+Y)=EX^2-E^2X+EY^2-E^2Y+2EXY-2EXEY\\
=\text{var}[X]+\text{var}[Y]+2\text{cov}[X,Y]$$

**Exercise 2.4:**

Simply using the Bayes rule here and we will get

$$P\{\text{got disease}|\text{test positive}\}=
\frac{P\{\text{test positive}|\text{got disease}\}P\{\text{got disease}\}}
{P\{\text{test positive}\}}=
\frac{0.99\times0.0001}{0.99\times0.0001+0.01\times0.9999}\approx0.01$$

**Exercise 2.5:**

This is just the calssical 3-doors problem, choose the other one is just right.

**Exercise 2.6:**

* a. ii. This can be easily verified using Bayes theorem.
* b. i and ii. Since the conditionally independent implies that
  $P(e_1,e_2|H)=P(e_1|H)P(e_2|H)$.

**Exercise 2.7:**

Here we can prove the statement by giving a contradiction.
Let $\Omega=\{a,b,c,d\}$ be a probability space containing four contradict
events with the same probability.
Then we can difine three different events upon $\Omega$, say
$A=\{a,b\},B=\{a,c\},C=\{a,d\}$.
They all have a probability of 0.5, and we can easily verify that they are pairwise
independent between all pairs of variables. However,

$$p(X_{1:n})=P(A,B,C)=\frac14\neq\frac18=P(A)P(B)P(C)$$
