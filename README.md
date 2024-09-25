# SimpleConvNet
CIFAR-10 classification seems to be a solved problem by now when it comes to
accuracy. However, there are still tasks, for example robust classification,
that we still don't know how to do, even on simple datasets like CIFAR-10.
Therefore, we think it is important to have simple models that perfrom well on
CIFAR-10, in order to build on them and use them as baselines for different
tasks.

There has been great work on creating simple models that can be trained quickly
and reach good performance on CIFAR-10. David Page has written a blog post
series how to train a model on CIFAR-10 in second. See for example
[this colab](https://colab.research.google.com/github/davidcpage/cifar10-fast/blob/master/bag_of_tricks.ipynb).
(The blog posts does not seem to be available anymore.)

Based on this, Johan Wind has written a blog post
["94% on CIFAR-10 in 94 lines and 94 seconds"](https://johanwind.github.io/2022/12/28/cifar_94.html)
where he simplifies the model and the training code.

Building on his code, we tried to simplify the model and the code a bit 
further, removing things such as label smoothing or weight decay. Furthermore,
we used a validation set to tune the hyperparameters.

Our model get to about 93.5% accuracy on the CIFAR-10 validation set in 24 
epochs.
We provide code for the [model](simple_conv_net.py) as well as the 
[training code](train_model.py).

## Requirements
- PyTorch
- torchvision




