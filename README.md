# Image Classification with ResNets using PyTorch

To access the original paper on Deep Residual Learning - [click here](https://arxiv.org/pdf/1512.03385.pdf) 

![alt text](https://github.com/Aditib2409/PyTorch_ResNets_FashionMNIST/blob/main/batch_images.jpg)

## Introduction
Network depth plays a crucial role in working with especially challenging datasets like ImageNet, Fashion MNIST, and Deep Convolution Neural Networks (CNN) have proven to lead to astonishing results in classification problems. These deeper layers in the CNN capture the low/mid/high level features and integrate them well enough to classify with improved accuracy. But, on the contrary, with more layers, the problem of vanishing/exploding gradients, which hamper the convergence of the training accuracy since the beginning of the training process have made CNN's a trouble. Hence, deeper networks necessarily do not always lead to a positive result. Though the solution to this problem, demanded Layer Normalization at the initial as well the intermediate layers. But another problem of degradation was still unsolved. Degradation was associated with the saturation of the accuracy and it's rapid reduction with training. In order to address this issue, the concept of Residual Networks was first introduced by Microsoft. 

Originally, the non-linear layer mapping H(x) had to be fit to an identity mapping which to an extreme was not optimal. With Deep Residual Networks, the stacked non-linear layers fit the mapping F(x):= H(x) - x. This residual mapping could easily be mapped to zero with every training. 

## Formulation of the Residual Mapping F(x) - 
The Residual architecture employs the use of shortcut connections that are simple identity maps that neither add any extra parameters nor lead to higher computational complexity. 

Precondition using the identity mapping as the optimal function. Since, finding perturbations to identity function can be easy reference than to learn the function as a new one altogether, The above formulation of residual mappings can be quickly justified.

### Residual Block

![Fig 1. Residual Block. Source of the image - "Deep Residual Learning for Image Recognition" by Kaiming et.al.,](https://github.com/Aditib2409/PyTorch_ResNets_FashionMNIST/blob/main/Screen%20Shot%202021-12-31%20at%205.16.42%20PM.png)

y = F(x, {Wi}) + x
Fig1. depicts a simple 2-layer residua block with ReLU activation function. The residual mapping which is the non-linear layer mapping is the output of this block. With multiple blocks stacked together, the changing input or output channels can result in dimension mismatch. And to cater to this challenge, projections (square projection matrix) by the shortcut channels were introduced in the paper. 

## Network Architecture - 
The network consists of a basic Plain Network, incorporated with the Residual Network. To start of, we begin with the introduction of the Plain Network.

### Plain Network
This network consists of layers of convolution layers with filter sizes as 3x3. The no. of filters is same as the output of the feature map size in the previous layer. And in instances, when the feature map size is halved, then the no.of filters is doubled to preserve the time complexity per layer. Each of these layers have a stride of 2 and the network finally ends with a Global average Pooling. The network has a total of 34 weighted layers.

### Residual Network
This part of the model uses the shortcut connections. The residual network in ResNet50 has a bottleneck design. For each residual function F, there are 3 layers instead of 2. The first layer is a simple 1x1 convolution layer followed by a 3x3 convolution layer which is the primary bottleneck and this is followed by another 1x1 convolution layer. The ResNet50 model can be evaluated as a ResNet34 but with each of the 34 weighted layer block having a 3-layer bottleneck design.

## Dataset - 
In this article, the ResNet50 model was implemented to classify the Fashion MNIST dataset. The model was pretrained and the intuition of pretraining the model asserted that the initial layers do not need to be trained over and over again during each epoch. Only the added/modified layers require training. Hence, the freezing process is implemented while pretraining the models. 

Additionally the pretrained models are usually trained on large amounts of data and use resources that aren't usually available to everyone. For instance, the ImageNet has around 1.2 million images with around 1000 classes.
The original ResNet50 model was trained on large datasets with 1000 classes but since the dataset considered here has only 10 classes, the final linear layer was modified before initiating the training phase.

The dataset is divided into a train_set and a test_set with 50,000 and 10,000 datasets respectively. The training was conducted at a learning rate of 0.001 with Adams optimizer and the training accuracy along with the test accuracy was calculated. To compute the loss for back propagation phase, the crossentropy function
was employed which inherently produced a softmax output and hence, an additonal softmax function was not explicitly mentioned here.

## Results of the pretrained model -

For optimized hyperparameters, such as batch size and learning rate, the complete dataset was trained for a few independent pairs before initiating the actual training and testing of the model. The training accuracy of the model was plotted as follows:

[!alt text](https://github.com/Aditib2409/PyTorch_ResNets_FashionMNIST/blob/main/Screen%20Shot%202022-01-01%20at%201.56.11%20AM.png)

The final test accuracy for the model and the corresponding dataset was around 90.35%. 




