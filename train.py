import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch import nn
from torch.utils.data import DataLoader
from cnn import CNNNetwork


BATCH_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 0.001



#GTZAN Dataset for genre classification...it included the corresponding Mel Spectrograms to be used by CNN
Train_DIR = "D:/Current_Focus/Music_Genre_Classification/Data/images_original"


def get_me_dataset():
    dataset = ImageFolder(Train_DIR,transform = transforms.Compose([
    transforms.Resize((150,150)),transforms.ToTensor()
    ]))
    return dataset


def get_me_trainloader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_dataloader



def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    z=0
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)
        
        print("Target")
        input.unsqueeze(0)
        print(input.shape)
        z=z+1
        print(z)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":
        
    device="cpu"
    
    
    print(f"Using {device}")
    
    train_data = get_me_dataset()
    train_dataloader = get_me_trainloader(train_data, BATCH_SIZE)
    

    # construct model and assign it to device
    cnn = CNNNetwork().to(device)
    print(cnn)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(cnn.state_dict(), "feedforwardnet.pth")