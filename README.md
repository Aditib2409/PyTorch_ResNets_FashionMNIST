# Image Classification with ResNets using PyTorch

https://www.icloud.com/iclouddrive/09d07Nk2MQj1lpSXsL0byBavA#batch_images


### Dataset - 
Downloaded and trained from the datasets provided by torchvision python module.
The dataset was divided into train and test sets with number of data points in each as 50,000 and 10,000 respectively. The training set was shuffled before
the training process.

### Model and Hyperparameters -
A pretrained ResNet50 model was downloaded and the model was used to train the training dataset with learning rate = 0.001 for about 10 epochs with a batch size
of 100. These hyperparameters were optimized on running the training algorithm multiple times for different sets of batch sizes and learning rates. 

### Optimizers
Both Adam and SGD optimizers were compared and the best optimizer was used during the loss (CrossEntropy) and backward propagation.

