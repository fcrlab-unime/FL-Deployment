import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from datasets import load_from_disk
import logging

# Imports for reproducibility
import random
import numpy as np

logging.basicConfig(level=logging.INFO)

class CNN(nn.Module):
    """A self-contained model class with its own training, testing, and data logic."""

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def get_weights(self):
        """Get model weights as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_weights(self, parameters):
        """Set model weights from a list of NumPy arrays."""
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.load_state_dict(state_dict, strict=True)

    @staticmethod
    def set_seed(seed: int):
        """Set all seeds to make results reproducible."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logging.info(f"Seeds set to {seed} for reproducibility.")

    @staticmethod
    def load_data(path: str, batch_size: int):
        """Load a dataset from disk and create dataloaders."""
        partition_train_test = load_from_disk(path)
        pytorch_transforms = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

        def apply_transforms(batch):
            batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
            return batch

        partition_train_test = partition_train_test.with_transform(apply_transforms)
        trainloader = DataLoader(
            partition_train_test["train"], batch_size=batch_size, shuffle=True
        )
        testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
        return trainloader, testloader

    def train_epoch(self, trainloader, epochs, learning_rate, device):
        """Train the model on the training set for a number of epochs."""
        self.to(device)
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        self.train()
        for _ in range(epochs):
            for batch in trainloader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                optimizer.zero_grad()
                loss = criterion(self(images), labels)
                loss.backward()
                optimizer.step()

    def test(self, testloader, device):
        """Validate the model on the test set."""
        self.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        self.eval()
        with torch.no_grad():
            for batch in testloader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                outputs = self(images)
                loss += criterion(outputs, labels).item()
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        
        accuracy = correct / total
        avg_loss = loss / len(testloader)
        return avg_loss, accuracy