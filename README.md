# Jax Saliency Methods

This repository demonstrates a suite of class saliency methods implemented using Jax and Flax. The primary focus is on how the algorithms work and how to implement them using the tools provided by Jax and Flax. MNIST was chosen as the dataset for the demonstrations because it doesn't require training complex models to perform well on the dataset. This allows us to use  simple perceptron or convolutional classification models and direct most of the focus on the saliency algorithms. In order to provide a better understanding of the algorithms values compared to other algorithms, brief descriptions of the algorithms features and drawbacks are provided in the section below. If you would like to understand the algorithms in more detail, the original papers are provided in the references section at the bottom.   

## Algorithms Summary

### Gradients

#### Requirements:
 - Classification Model: Any continuously differentiable model.
 - Gradients of the input.

#### Pros:
 - Easy to implement.
 - Simple to understand.
 - High resolution saliency map.
 - Fast: Only needs a single backwards pass to calculate gradients.
 - Works on a wide range of models.

#### Cons:
 - Lots of noise in the saliency map.
 - Low accuracy.
 - Violates the sensitivity axiom (gradient can be zero despite of the input changing), shown in [Integrated Gradients].

### Gradients $\times$ Input

#### Requirements:
 - Same as Gradients method

#### Pros:
 - Same as Gradients method, but better accuracy.

#### Cons:
 - Same as Gradients method

### Integrated Gradients

#### Requirements:
 - Classification Model: Any continuously differentiable model.
 - Integral of the gradients of multiple scaled inputs. 

#### Pros:
 - High resolution saliency map.
 - Works on a wide range of models.
 - Class discriminative (localizes the category in the image).

#### Cons:
 - Slow: Requires multiple backwards passes of the model.
 - Complex: Requires more knowledge and code than gradients.

### Deconvolution

#### Requirements:
 - Classification Model: A convolutional neural network.
 - A deconvolutional version of the classification model.
 - Access the values of a convolutional layer in the classification model.

#### Pros:
 - High resolution saliency map.
 - Fast: Need a single pass through the deconvolutional model.

#### Cons:
 - Need to create a deconvolutional model.
 - Fails to highlight gradients that contribute negatively, as described in [DeepLIFT].
 - Not class discriminative (doesn't localize the category in the image), shown in [Grad-CAM].

### Guided Backpropagation

#### Requirements:
 - Classification Model: A convolutional neural network.
 - A new relu layer, called guided relu.
 - Gradients of the inputs with respect to the model with guided relu layers.

#### Pros:
 - High resolution saliency map.
 - Fast: Need a single backward pass through the model with guided relu layers.

#### Cons:
 - Need to create a new type of relu layer.
 - Guided relu performs poorly during training.
 - Fails to highlight gradients that contribute negatively, as described in [DeepLIFT].
 - Complex: Implementing guided relu requires an advanced understanding of the frameworks tools.
 - Not class discriminative (doesn't localize the category in the image), shown in [Grad-CAM].

### Class Activation Mapping (CAM)

#### Requirements:
 - Classification Model: A convolutional neural network.
 - Access the values of a convolutional layer in the classification model.

#### Pros:
 - Fast: Only need a mean and dot product operation on the convolutional layer.
 - Simple to understand.
 - Class discriminative (localizes the category in the image), shown in [Grad-CAM].

#### Cons:
 - Low resolution saliency map.
 - Global average pooling performs poorly during training (but there is a way to train without it).

### Grad-CAM

#### Requirements:
 - Classification Model: A convolutional neural network.
 - Access the values and gradients of the final convolutional layer in the classification model.

#### Pros:
 - Fast: Need to perform a single backwards pass, plus the same CAM operations.
 - Class discriminative (localizes the category in the image), shown in [Grad-CAM].

#### Cons:
 - Low resolution saliency map.
 - Requires more code and complexity than CAM while providing similar accuracy.


### Guided Grad-CAM

#### Requirements:
 - Classification Model: A convolutional neural network.
 - All the requirements from Grad-CAM and Guided backprop.

#### Pros:
 - Fast: Only a single backwards pass is needed.
 - Class discriminative (localizes the category in the image), shown in [Grad-CAM].
 - High resolution saliency map.

#### Cons:
 - Complex: Need to implement both Grad-CAM and guided backprop.


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
 - [DeepLIFT] - [Learning Important Features Through Propagating Activation Differences](https://arxiv.org/abs/1704.02685)

