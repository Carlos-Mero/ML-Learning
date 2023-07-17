# MML-Part2

> Central Machine Learning Problems

## When Models Meet Data

Here are a set of basic terminologies used in the domain of machine learning.
Some of them need additional attention thongh.

**Empirical Risk Minimization(ERM)** <-> 经验风险最小化

This method says that since we can not easily find out the risk of the
coming future, we can turn back to the empirical things and apply a
optimization algorithm to gain the minimal empirical risk.

If we want to further model our problems using probability distributions,
then instead of using loss function for optimization, we'll need to use a
similar concept likelihood, and using the **Maximum Likelihood Estimation(MLE)**
<-> 极大似然估计.

In probabilistic models, we'll consider both the variable and the parameters
as random variables, thus we can ask in the given model, what do the parameters
most likely to be?
This is the central idea of the MLE.
