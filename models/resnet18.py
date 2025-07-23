# For use with the CIFAR-10 dataset.

import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.models import resnet18
from datasets import load_from_disk

class RESNET18(nn.Module):
    """A self-contained ResNet-18 model class for CIFAR-10."""

    def __init__(self):
        super(RESNET18, self).__init__()
        # Instantiate the ResNet-18 model from torchvision
        self.model = resnet18(weights=None) # From scratch
        
        # Adapt the final fully connected layer for 10 classes (CIFAR-10)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward pass of the model."""
        return self.model(x)

    def get_weights(self):
        """Get model weights as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_weights(self, parameters):
        """Set model weights from a list of NumPy arrays."""
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.load_state_dict(state_dict, strict=True)

    @staticmethod
    def load_data(path: str, batch_size: int):
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

    def train_epoch(self, trainloader, epochs, learning_rate, device):
        """Train the model on the training set for a number of epochs."""
        self.to(device)
        criterion = torch.nn.CrossEntropyLoss().to(device)
        # Using Adam optimizer as in the original script
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.train()
        for _ in range(epochs):
            for batch in trainloader:
                # Use 'img' key for CIFAR-10
                images = batch["img"].to(device)
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
                # Use 'img' key for CIFAR-10
                images = batch["img"].to(device)
                labels = batch["label"].to(device)
                outputs = self(images)
                loss += criterion(outputs, labels).item()
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        
        accuracy = correct / total
        avg_loss = loss / len(testloader)
        return avg_loss, accuracy
