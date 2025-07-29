import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# NVFlare Client API
from nvflare.client import flare

# Import the model definition
from .net import Net

# Constants
DATA_PATH = "./data"
BATCH_SIZE = 64
LEARNING_RATE = 0.01
MOMENTUM = 0.9
NUM_LOCAL_EPOCHS = 1

def get_dataloader():
    """Prepares the Fashion-MNIST dataset."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_dataset = datasets.FashionMNIST(
        root=DATA_PATH, train=True, download=True, transform=transform
    )
    return DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

def train(model, dataloader, optimizer, loss_fn, device):
    """Performs one epoch of local training."""
    model.train()
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

def main():
    # Initialize the NVFlare Client API
    flare.init()

    # Setup model, data, and training components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    train_loader = get_dataloader()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    loss_fn = nn.CrossEntropyLoss()

    # The model on the client is initialized from the server,
    # so we don't need to load any initial weights.

    print("--- NVFlare Client API Initialized ---")
    
    # Get the number of rounds from the flare client
    num_rounds = flare.get_total_rounds()

    # Main training loop
    for current_round in range(num_rounds):
        print(f"\n--- Round {current_round + 1}/{num_rounds} ---")

        # Receive model from the server
        input_model = flare.receive()
        
        # Load the received model weights
        model.load_state_dict(input_model.params)
        print("Received model from server.")
        
        # Perform local training for N epochs
        for epoch in range(NUM_LOCAL_EPOCHS):
            train(model, train_loader, optimizer, loss_fn, device)
            print(f"Finished local epoch {epoch + 1}/{NUM_LOCAL_EPOCHS}")
        
        # Create the output model to send back to the server
        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            metrics={"train_loss": 1.0}, # Example metric
            meta={"NUM_STEPS_CURRENT_ROUND": len(train_loader)} # Example metadata
        )
        
        # Send the updated model to the server
        flare.send(output_model)
        print("Sent updated model to server.")

if __name__ == "__main__":
    main()