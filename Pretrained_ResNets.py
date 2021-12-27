import math
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn
import matplotlib.pyplot as plt
import itertools
from torch.utils.tensorboard import SummaryWriter


torch.set_printoptions(linewidth=120)

train_set = torchvision.datasets.FashionMNIST(
	root = './data/FashionMNIST'
	,train=True
	,download=True  # we achieved the EXTRACTION process
	,transform=transforms.Compose([
		transforms.ToTensor() # we achieved the Tranformation into Tensor Datastructure.
	])
)

batch_size = 100
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size) 

#optimizer = optim.SGD(network.parameters(), lr=0.01)

#######  CREATING THE CNN ######

class FashionMNISTResNet(nn.Module):
	def __init__(self, in_channels=1):
		super(FashionMNISTResNet, self).__init__()
		# loading a pretrained model
		self.model = torchvision.models.resnet50(pretrained=True)
		# changing the input color channels to 1 since original resnet has 3 channels for RGB
		self.model.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)

		#change the output layer to 10 ckasses as the original resnet has 1000 classes
		num_ftrs = self.model.fc.in_features
		self.model.fc = nn.Linear(num_ftrs, 10)

#### FORWARD PROPAGATION #######
	def forward(self, t):
		
		return self.model(t)


def get_num_correct_predictions(preds, labels):
	return preds.argmax(dim=1).eq(labels).sum().item()


#batch_size_list = [100, 1000, 10000]
lr_list = [0.01, 0.001, 0.0001]

acc = []
fin_acc = []
ep = []


#for batch_size in batch_size_list:
for lr in lr_list:

		network = FashionMNISTResNet()

		optimizer = optim.Adam(network.parameters(), lr=lr)

		#images, labels = next(iter(train_loader)) This did'nt work


		#### TRAINING WITH MULTIPLE EPOCHS: THE COMPLETE TRAINING LOOP 
		for epoch in range(5):
		######  TRAINING WITH A BATCH ######

			total_loss = 0
			total_correct = 0
			ep.append(epoch)
			for batch in train_loader:
				images, labels = batch 
				preds = network(images) 
				loss = F.cross_entropy(preds, labels)
				optimizer.zero_grad()
				loss.backward() # calculating the gradients
				optimizer.step() # updating the weights

				total_loss += loss.item() * batch_size # to compare runs for different batch_sizes
				total_correct += get_num_correct_predictions(preds, labels)
				"""
				tb.add_scalar('Loss', total_loss, epoch)
				tb.add_scalar('Number Correct', total_correct, epoch)
				tb.add_scalar('Accuracy', total_correct / len(train_set), epoch)

				for name, weight in network.named_parameters():
					tb.add_histogram(name, weight, epoch)
					tb.add_histogram(f'{name}.grad', weight.grad, epoch)
				"""
			accuracy = total_correct / len(train_set) * 100
			acc.append(accuracy)
			print("epoch:", epoch, "Learning rate:", lr, "total_correct:", total_correct, "loss:", total_loss, "Accuracy:", accuracy,'%')
		fin_acc.append(accuracy)
		print("Learning_rate:", lr, "Accuracy:", accuracy)












