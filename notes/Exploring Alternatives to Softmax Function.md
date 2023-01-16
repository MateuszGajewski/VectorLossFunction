# Source
https://arxiv.org/pdf/2011.11538.pdf

# Key points

- Softmax function is a popular choice in deep learning classification tasks, where it typically appears as the last layer. Recently,  this function has found application in other operations as well, such as the attention mechanism.
- In the articels alternatives to softmax function are explored and compared.

# Explored alternatives

## Taylor Softmax

- $Tsm(z)_i = \frac{1+z_i+0.5z_i^2}{\sum_{j=1}^K 1 + z_j + 0.5z_j^2}$
- Second order Tayor series approximation for $e^z = 1+z+0.5z^2$
- It is always positive - so the function generates propability distribution.

## Soft-margin softmax

- Soft-margin (SM) softmax reduces intra-class distances but enhances inter-class discrimination, by introducing a distance margin into the logits.
- $SMsm(z)_i = \frac{e^{z_i-m}}{\sum_{i \neq j}^{K}e^{z_j} + e^{z_i-m}}$ - only in training phase

## SM-Taylor softmax
- SM-Taylor softmax uses the same formula as given in equation in $SMsm(z)_i = \frac{e^{z_i-m}}{\sum_{i \neq j}^{K}e^{z_j} + e^{z_i-m}}$ while using approximation of $e^z$ using taylor function

# Comparison 
Image classification task on MNIST, CIFAR10 and CIFAR100 datasets, was performedusing the softmax function and its various alternatives. The goal was not to reach the state of the art accuracy for each dataset but to compare the influence of each alternative. Therefore, reasonably sized standard neural network architectures with no ensembling and no data augmentation were used.

## Results

The best score at each dataset was achieved by SM-Taylor softmax, but all of proposed alternatives performed better than softmax (but some only by a very small margin).

Apart of scores, another advantage is fluctuation in the training loss for the softmax function,  whereas the plot is comparatively smoother for all its alternatives(but it is visible only in MNIST dataset).

$Note$ : There are no confidence intervals - we are not sure if all changes are meaningfull.
