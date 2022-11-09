# Jax Saliency Methods

This repository demonstrates a suite of class saliency methods implemented using Jax and Flax. The primary focus is on how the algorithms work and how to implement them using the tools provided by Jax and Flax. MNIST was chosen as the dataset for the demonstrations because it doesn't require training complex models to perform well on the dataset. This allows us to use  simple perceptron or convolutional classification models and direct most of the focus on the saliency algorithms. In order to provide a better understanding of the algorithms values compared to other algorithms, brief descriptions of the algorithms features and drawbacks are provided in the section below. If you would like to understand the algorithms in more detail, the original papers are provided in the references section at the bottom.   

## Algorithms Summary

### Gradients


### Gradients $\times$ Input


### Integrated Gradients



### Deconvolution



### Guided Backpropagation



### Class Activation Mapping (CAM)


### Grad-CAM



## Installation Requirements

Jax is a requirement for the notebooks, you can install GPU accelerated Jax by running the command below.
```
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
``` 
The latest Flax can be installed with the following command.
```
pip install -q clu ml-collections git+https://github.com/google/flax
``` 
The following packages are also required.
```
pip install jupyter
pip install pandas
pip install matplotlib
```


## References
 - [Gradients] - [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034)
 - [Integrated Gradients] - [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365)
 - [Deconvolution] - [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901)
 - [Guided Backprop] - [Striving for Simplicity: The All Convolutional Net](https://arxiv.org/abs/1412.6806)
 - [CAM] - [Learning Deep Features for Discriminative Localization](https://arxiv.org/abs/1512.04150)
 - [Grad-CAM] - [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)


