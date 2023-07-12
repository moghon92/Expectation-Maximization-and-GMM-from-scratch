This is an implementation of E
xpectation Maximizaton and Gussian Mixture model from scratch.

##  Expectation Maximization (EM)
 is an iterative algorithm used to estimate the parameters of statistical models, particularly when dealing with incomplete or missing data. It is commonly employed in situations where there are hidden or unobserved variables that affect the data generation process. The EM algorithm consists of two main steps: the E-step (Expectation step) and the M-step (Maximization step).

### In the E-step
 the algorithm starts with an initial estimation of the model parameters. It then computes the expected values of the hidden variables, given the observed data and the current parameter estimates. This step involves computing the probability or likelihood of the hidden variables given the observed data, using the current parameter estimates.

### In the M-step
 the algorithm updates the parameter estimates by maximizing the expected log-likelihood computed in the E-step. This step involves finding the values of the parameters that maximize the expected log-likelihood, given the observed data and the expected values of the hidden variables computed in the E-step.

The algorithm iteratively alternates between the E-step and M-step until convergence is reached, which is typically determined by a predefined stopping criterion such as the change in parameter estimates or the log-likelihood.


# Gaussian Mixture Model (GMM)
 is a probabilistic model that represents a probability distribution as a combination of multiple Gaussian (normal) distributions. It is commonly used for modeling complex data with multiple underlying subpopulations or clusters
 
 ## The key steps in using GMM are:

- Initialization: Initialize the GMM by specifying the number of components (clusters) and randomly or heuristically initializing the mean vectors, covariance matrices, and component weights.

- E-step (Expectation step): Compute the probabilities or responsibilities of each component for each data point. This is done by calculating the posterior probabilities of the component assignment variable given the observed data and current parameter estimates. The responsibilities represent the degree of certainty that each component generated each data point.

- M-step (Maximization step): Update the parameters of the Gaussian components based on the responsibilities calculated in the E-step. Specifically, update the mean vectors, covariance matrices, and component weights by maximizing the expected log-likelihood of the data. This step involves solving optimization problems to find the parameter values that maximize the objective function.

- Iteration: Repeat the E-step and M-step iteratively until convergence is reached. In each iteration, the responsibilities are recalculated based on the updated parameters, and the parameters are refined based on the responsibilities.

- Output: Once the algorithm converges, the estimated parameters of the GMM can be used for various tasks, such as clustering the data points based on the most likely component assignments, generating new data points from the learned distribution, or using the GMM as a generative model for further analysis