# Source
https://arxiv.org/pdf/1511.05042.pdf?ref=https://githubhelp.com

#  Key points
Exploring alternatives to softmax function belonging to spherical softmax family:
- Spherical Softmax
- Taylor Sotmax
# Explored alternatives
## Spherical Family
#### Definition
Familiy of sperical functions are functions which can be expressed as: 
$L(s,q,z_c,y_c)$, where:
- $s=\sum_iz_i$
- $q = \mid \mid z\mid \mid ^2$
-  $o_c$ - logit of true class
- $y_c$ - true class
#### Properties
Has an upper bound (details in paper) and therefor there is an efficient alghorithm in $O(d^2)$

### Spherical Softmax
$SSsm(z)_i = \frac{z_i^2 + \epsilon}{\sum_{i}z_j^2 + \epsilon}$, where $\epsilon$ is a small constant added for numerical stability.
- Unvariant to rescaling - in contra to softmax (invariant to translation) - it is unclear wich one is better
- gradient: $\frac{\partial L}{\partial o_c} = \frac{2o_c}{\sum_{i=1}^D(o_i^2+\epsilon)} - \frac{2o_c}{o_c^2+\epsilon}$ and $\frac{\partial L}{\partial o_{k \neq c}} = \frac{2o_c}{\sum_{i=1}^D(o_i^2+\epsilon)} - \frac{2o_c}{o_c^2+\epsilon}$ where $c$ is target class

### Taylor Softmax
- second-order Taylor expansion of the exponential around zero
- $TSm(o)_k = \frac{1+o_k+0.5o_k^2}{\sum_{i=1}^D(1+o_i+0.5o_i^2)}$
- gradient $\frac{\partial L}{\partial o_c} = \frac{1+o_c}{\sum_{i=1}^D(1+o_i+0.5o_i^2)} - \frac{1+o_c}{1+o_i+0.5o_i^2}$ and $\frac{\partial L}{\partial o_{k \neq c}} = \frac{1+o_k}{\sum_{i=1}^D(1+o_i+0.5o_i^2)}$
- The gradients are well-behaved as well, wi
- 
- th no risk of numerical instability. Therefore, contrary to the spherical softmax, we do not need to use the extra hyperparameter $\epsilon$.
- Furthermore, unlike the spherical softmax, the Taylor softmax has a small asymmetry around zero.

# Experiments 
- On MNIST and CIFAR10, the spherical losses work surprisingly well and, for the fixed architectures and they even outperform the log-softmax. 
- This suggests that the log-softmax is not necessarily the best loss function for classification and that alternatives such as categorical log-losses from the spherical family might be preferred in a broad range of applications.
- On the other hand, in our experiments with higher output dimensions, i.e. on CIFAR100, the Penntree bank and the one Billion Word dataset, we found that the log softmax yields better results than the log-spherical softmax and the log-Taylor softmax.
