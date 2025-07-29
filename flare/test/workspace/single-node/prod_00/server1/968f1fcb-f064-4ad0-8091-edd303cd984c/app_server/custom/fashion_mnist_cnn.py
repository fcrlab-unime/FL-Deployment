# File: jobs/fashion_mnist_cnn/app/custom/fashion_mnist_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import nvflare.client as flare


class SimpleCNN(nn.Module):
    """Simple CNN for Fashion-MNIST classification"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, 128 * 3 * 3)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


def get_data_loaders(data_path="./data", batch_size=32, site_name="site-1"):
    """Create data loaders for Fashion-MNIST with data partitioning"""
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST normalization
    ])
    
    # Load Fashion-MNIST dataset
    train_dataset = datasets.FashionMNIST(
        data_path, train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        data_path, train=False, download=True, transform=transform
    )
    
    # Simple data partitioning based on site name
    # In practice, you'd implement more sophisticated partitioning
    site_num = int(site_name.split('-')[-1]) if 'site-' in site_name else 1
    num_sites = 3  # Assuming 3 clients
    
    # Partition training data
    total_train = len(train_dataset)
    samples_per_site = total_train // num_sites
    start_idx = (site_num - 1) * samples_per_site
    end_idx = start_idx + samples_per_site if site_num < num_sites else total_train
    
    train_indices = list(range(start_idx, end_idx))
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_model(model, train_loader, device, epochs=1, lr=0.001):
    """Train the model for specified epochs"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    
    total_loss = 0.0
    total_samples = 0
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, '
                      f'Loss: {loss.item():.6f}')
    
    avg_loss = total_loss / total_samples
    return avg_loss


def evaluate_model(model, test_loader, device):
    """Evaluate the model on test data"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    test_loss /= total
    accuracy = 100. * correct / total
    
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return test_loss, accuracy


def main():
    """Main federated learning training loop"""
    # Initialize FLARE client
    flare.init()
    
    # Get system info
    site_name = flare.get_site_name()
    print(f"Starting training on site: {site_name}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    train_loader, test_loader = get_data_loaders(site_name=site_name)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Initialize model
    model = SimpleCNN(num_classes=10).to(device)
    
    # Federated learning loop
    while flare.is_running():
        # Receive global model from server
        input_model = flare.receive()
        print(f"Received global model from server")
        
        # Update local model with global parameters
        if input_model:
            model.load_state_dict(input_model.params)
        
        # Local training
        print("Starting local training...")
        train_loss = train_model(
            model, train_loader, device, 
            epochs=2, lr=0.001
        )
        
        # Local evaluation
        test_loss, accuracy = evaluate_model(model, test_loader, device)
        
        # Prepare metrics to send back
        metrics = {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "accuracy": accuracy,
            "num_samples": len(train_loader.dataset)
        }
        
        # Send updated model back to server
        flare.send(flare.FLModel(
            params=model.state_dict(),
            metrics=metrics,
            meta={"site_name": site_name}
        ))
        
        print(f"Sent model update to server. Train loss: {train_loss:.4f}, "
              f"Test accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()