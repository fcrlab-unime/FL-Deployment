import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import importlib
import logging

logging.basicConfig(level=logging.INFO)

# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, learning_rate):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        self.net.set_weights(parameters)
        
        # Train the model
        self.net.train_epoch(
            self.trainloader,
            self.local_epochs,
            self.lr,
            self.device,
        )

        # Evaluate the model to get the validation loss and accuracy
        loss, accuracy = self.net.test(self.valloader, self.device)
        
        results = {
            "val_loss": loss,
            "val_accuracy": accuracy,
        }
        
        return self.net.get_weights(), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        self.net.set_weights(parameters)
        loss, accuracy = self.net.test(self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # --- Start of new/modified code ---

    # Get the seed from the server's run config for reproducibility
    # Use a default seed if not provided
    seed = context.run_config.get("seed", 42)
    logging.info(f"Using seed: {seed}")

    # Get model and dataset name from the server's run config
    model_name = context.run_config["model"] # e.g., "cnn", "dnn", "resnet18"
    
    # Dynamically import the model class from the correct file
    try:
        module = importlib.import_module(f"models.{model_name}")
        Net = getattr(module, model_name.upper()) 
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not load model '{model_name.upper()}' from 'models/{model_name}.py'") from e

    # Set the seed for the entire client using the static method
    Net.set_seed(seed)
    
    # --- End of new/modified code ---

    # Read client-specific configuration
    num_partitions = context.run_config["num-partitions"]
    dataset_name = context.run_config["dataset-name"]
    client_n = context.node_config["client-n"]
    
    dataset_path = f"/app/dataset/{dataset_name}/{num_partitions}_partitions/{dataset_name}_part_{client_n}"
    logging.info(dataset_path)

    # Read run-specific hyperparameters
    batch_size = context.run_config["batch-size"]
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]
    
    # Load the data using the static method from the imported model class
    # Seeding is done, so this will now be reproducible
    trainloader, valloader = Net.load_data(dataset_path, batch_size)

    # Instantiate the model. Weight initialization will be reproducible.
    net = Net()

    # Return a FlowerClient instance
    return FlowerClient(net, trainloader, valloader, local_epochs, learning_rate).to_client()

# Flower ClientApp
app = ClientApp(client_fn)
