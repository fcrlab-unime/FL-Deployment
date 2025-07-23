# For use with the CIFAR-10 dataset.

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.models import resnet18
from datasets import load_from_disk

# 1. MODEL
def Net():
    """Wrapper for ResNet-18 to be used as Net."""
    model = resnet18(weights=None) # From scratch
    # Adapt the final layer for 10 classes (CIFAR-10)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    return model

# 2. WEIGHTS MANIPULATION
def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

# 3. DATA LOADING
def load_data_from_disk(path: str, batch_size: int):
    """Load CIFAR-10 from disk and create DataLoaders."""
    partition_train_test = load_from_disk(path)
    # Transforms for color 32x32 images
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        # The key for CIFAR-10 in Hugging Face is 'img', not 'image'
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader

# 4. TRAINING & TESTING
def train(net, trainloader, valloader, epochs, learning_rate, device):
    """Train the model on the training set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            # Use 'img' key for CIFAR-10
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            criterion(net(images), labels).backward()
            optimizer.step()
    val_loss, val_acc = test(net, valloader, device)
    return {"val_loss": val_loss, "val_accuracy": val_acc}

def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            # Use 'img' key for CIFAR-10
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss / len(testloader), accuracy