# Channel normalization in convolutional neural networks

This folder provides the code for reproducing the results in the paper: 

**``Channel Normalization in Convolutional Neural Network avoids Vanishing Gradients''**, by Zhenwei Dai and Reinhard Heckel, ICML workshop 2019.

The paper is available online [[here]](http://www.reinhardheckel.com/papers/channel_normalization.pdf).

## Installation

The code is written in python and relies on pytorch. The following libraries are required: 
- python 3
- pytorch
- numpy
- skimage
- matplotlib
- scikit-image
- jupyter

## Citation
```
@InProceedings{dai_channel_2019,
    author    = {Zhenwai Dai and Reinhard Heckel},
    title     = {Channel Normalization in Convolutional Neural Network avoids Vanishing Gradients},
    booktitle   = {International Conference on Machine Learning, Deep Phenomena Workshop},
    year      = {2019}
}
```

## Content of the repository

**one_dim_net_convergence_paper.ipynb** includes the code to run gradient descent on deep decoder, multi-channel CNN and linear CNN, and can be used to reproduce Figure 1,2, and 5.

**visualize_loss_function_landscape.ipynb** plots the loss function landscape of multi-channel CNN and linear CNN

**distribution_gradient_linear_network_initialization.ipynb** plots the gradients norm at initialization (with Normal distribution) for a linear CNN, to reproduced Figure 4a and 4b.

**distribution_gradient_CNN_initialization.ipynb** plots the gradients norm at initialization (with Normal distribution) of a multichannel CNN, to reproduced Figure 4c and 4d.

## Licence

All files are provided under the terms of the Apache License, Version 2.0.
