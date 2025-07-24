"""A Flower/PyTorch app with Prometheus metrics integration."""

from typing import Dict, List, Optional, Tuple, Union
import logging
logging.basicConfig(level=logging.INFO)

from flwr.common import (
    Context,
    EvaluateRes,
    Metrics,
    Scalar,
    ndarrays_to_parameters,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from prometheus_client import Gauge, start_http_server
import importlib


# --- Prometheus Metrics Definition ---
accuracy_gauge = Gauge(
    "model_accuracy", "Current accuracy of the global model."
)
loss_gauge = Gauge("model_loss", "Current loss of the global model.")


# --- Metric Aggregation Function ---
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Computes a weighted average of the given metrics."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


# --- Custom Strategy with Prometheus Integration ---
class PrometheusFedAvg(FedAvg):
    """Custom FedAvg strategy to update Prometheus gauges."""

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses and metrics, and update Prometheus gauges."""
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        if aggregated_loss is not None:
            loss_gauge.set(aggregated_loss)
            logging.info(f"Round {server_round}: Global loss set to {aggregated_loss:.4f}")

        if aggregated_metrics and "accuracy" in aggregated_metrics:
            accuracy = aggregated_metrics["accuracy"]
            if isinstance(accuracy, (float, int)):
                accuracy_gauge.set(accuracy)
                logging.info(f"Round {server_round}: Global accuracy set to {accuracy:.4f}")

        return aggregated_loss, aggregated_metrics


# --- ServerApp Factory Function ---
def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    model_name = context.run_config["model"]

    # --- Start of new/modified code ---

    # Get the seed from the run config for reproducibility
    seed = context.run_config.get("seed", 42)
    logging.info(f"Using seed: {seed} for server-side initialization")
    
    # Dynamically import the model class
    try:
        module = importlib.import_module(f"models.{model_name}")
        Net = getattr(module, model_name.upper()) 
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not load model '{model_name.upper()}' from 'models/{model_name}.py'") from e
    
    # Set the seed using the static method from the model file.
    # This ensures that the initial model parameters are generated deterministically.
    Net.set_seed(seed)

    # --- End of new/modified code ---

    # Initialize model parameters (now reproducible)
    ndarrays = Net().get_weights()
    parameters = ndarrays_to_parameters(ndarrays)

    # Define the custom strategy
    strategy = PrometheusFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=context.run_config["fraction-evaluate"],
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# --- Main Application ---
app = ServerApp(server_fn=server_fn)

@app.lifespan()
def lifespan(context: Context):
    """Lifespan event to start Prometheus metrics server."""
    start_http_server(8001)
    logging.info("Prometheus metrics server started on port 8001")
    yield
    logging.info("Shutting down Prometheus metrics server")