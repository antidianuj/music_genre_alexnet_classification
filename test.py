import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch import nn
from torch.utils.data import DataLoader
from cnn import CNNNetwork


model = torch.load("feedforwardnet.pth")

test_dir="D:/Current_Focus/Music_Genre_Classification/Test"

dataset=ImageFolder(test_dir,transform = transforms.Compose([
    transforms.Resize((150,150)),transforms.ToTensor()
    ]))


input,target=dataset[0];
input_o=input.unsqueeze(0)

prediction = model(input)