import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
import nvflare.client as flare
import os
import numpy as np
import random
from datasets import load_from_disk


def set_seed(seed=42):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

    def load_data(self, path: str, batch_size: int):
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
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.train()
        
        total_loss = 0.0
        total_samples = 0
        
        for epoch in range(epochs):
            for batch_idx, batch in enumerate(trainloader):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                optimizer.zero_grad()
                loss = criterion(self(images), labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)
                
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / total_samples
        return avg_loss

    def test(self, testloader, device):
        """Evaluate the model on the test set."""
        self.to(device)
        self.eval()
        criterion = torch.nn.CrossEntropyLoss()
        
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in testloader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                outputs = self(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_loss /= total
        accuracy = 100.0 * correct / total
        
        print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
        return test_loss, accuracy


def get_site_data_path(site_name: str, base_data_path: str = "/data/5_partitions") -> str:
    """Get the data path for a specific site based on site name."""
    # Extract site number from site name (e.g., "site-1" -> 1)
    if 'site-' in site_name:
        site_num = site_name.split('-')[-1]
    else:
        site_num = "1"  # default
    
    # Map site to partition directory
    partition_path = os.path.join(base_data_path, f"fashion_mnist_part_{site_num}")
    
    if not os.path.exists(partition_path):
        raise FileNotFoundError(f"Data partition not found at: {partition_path}")
    
    return partition_path


def main():
    """Main federated learning training loop using Client API with pre-partitioned data"""
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Initialize FLARE client
    flare.init()
    
    # Get system info
    site_name = flare.get_site_name()
    print(f"Starting training on site: {site_name}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = CNN()
    
    # Get site-specific data path
    try:
        data_path = get_site_data_path(site_name)
        print(f"Loading data from: {data_path}")
        
        # Load pre-partitioned data using the model's method
        trainloader, testloader = model.load_data(data_path, batch_size=32)
        print(f"Training samples: {len(trainloader.dataset)}")
        print(f"Test samples: {len(testloader.dataset)}")
        
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return
    
    # Federated learning loop
    while flare.is_running():
        # Set seed before each round for consistency
        set_seed(42)
        
        # Receive global model parameters from server
        input_model = flare.receive()
        print("Received global model from server")
        
        # Load global model parameters if available
        if input_model and input_model.params:
            model.load_state_dict(input_model.params)
            print("Updated local model with global parameters")
        
        # Perform local training
        print("Starting local training...")
        train_loss = model.train_epoch(
            trainloader, 
            epochs=2, 
            learning_rate=0.001, 
            device=device
        )
        
        # Perform local evaluation
        test_loss, accuracy = model.test(testloader, device)
        
        # Prepare metrics to send back
        metrics = {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "accuracy": accuracy,
            "num_samples": len(trainloader.dataset)
        }
        
        # Send updated model parameters back to server
        flare.send(flare.FLModel(
            params=model.state_dict(),
            metrics=metrics,
            meta={"site_name": site_name}
        ))
        
        print(f"Sent model update to server. Train loss: {train_loss:.4f}, "
              f"Test accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()