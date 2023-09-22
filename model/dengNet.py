import numpy as np
import torch
import torch.nn as nn
from .srm_filter_kernel import all_normalized_hpf_list




IMAGE_SIZE = 256
BATCH_SIZE = 32 // 2
EPOCHS = 200
#EPOCHS = 2
LR = 0.01
WEIGHT_DECAY = 5e-4
EMBEDDING_RATE = 0.4

# TRAIN_FILE_COUNT = 8000
TRAIN_PRINT_FREQUENCY = 100
EVAL_PRINT_FREQUENCY = 1
DECAY_EPOCH = [80, 140, 180]

#FINETUNE_EPOCHS = 1
FINETUNE_EPOCHS = 100


class TLU(nn.Module):
	def __init__(self, threshold):
		super(TLU, self).__init__()
		self.threshold = threshold

	
	def forward(self, input):
		output = torch.clamp(input, min=-self.threshold, max=self.threshold)

		return output





class HPF(nn.Module):
	def __init__(self):
		super(HPF, self).__init__()

		all_hpf_list_5x5 = []

		for hpf_item in all_normalized_hpf_list:
			if hpf_item.shape[0] == 3:
				hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')

			all_hpf_list_5x5.append(hpf_item)

		hpf_weight = nn.Parameter(torch.Tensor(all_hpf_list_5x5).view(30, 1, 5, 5), requires_grad=False)

		self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
		self.hpf.weight = hpf_weight
		self.tlu = TLU(3.0)


	def forward(self, input):
		output = self.hpf(input)
		output = self.tlu(output)

		return output




class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		self.group1 = HPF()

		self.group2 = nn.Sequential(
			nn.Conv2d(30, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),

			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),

			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),

			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),

			nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
		)

		self.group3 = nn.Sequential(
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),

			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),

			nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
		)

		self.group4 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),

			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),

			nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
		)

		self.group5 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),

			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(),

			nn.AvgPool2d(kernel_size=32, stride=1)
		)

		self.fc1 = nn.Linear(1 * 1 * 256, 2)

	def forward(self, input):
		output = input

		output = self.group1(output)
		output = self.group2(output)
		output = self.group3(output)
		output = self.group4(output)
		output = self.group5(output)

		output = output.view(output.size(0), -1)
		output = self.fc1(output)

		return output




