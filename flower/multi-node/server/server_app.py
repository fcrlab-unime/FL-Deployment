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

from model.model import Net, get_weights

# --- Prometheus Metrics Definition ---
# Define a gauge to track the global model accuracy
accuracy_gauge = Gauge(
    "model_accuracy", "Current accuracy of the global model."
)
# Define a gauge to track the global model loss
loss_gauge = Gauge("model_loss", "Current loss of the global model.")


# --- Metric Aggregation Function ---
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Computes a weighted average of the given metrics.

    This function is used by the strategy to aggregate the metrics sent by the
    clients.

    Args:
        metrics (List[Tuple[int, Metrics]]): A list of tuples, where each
            tuple contains the number of examples used for evaluation
            and the metrics dictionary.

    Returns:
        Metrics: A dictionary containing the aggregated metric (e.g.,
            {"accuracy": 0.95}).
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
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

        # Call the parent class method to perform the actual aggregation
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        # Update Prometheus gauges if the aggregated values are available
        if aggregated_loss is not None:
            # Set the loss gauge to the aggregated loss
            loss_gauge.set(aggregated_loss)
            logging.info(f"Round {server_round}: Global loss set to {aggregated_loss:.4f}")

        if aggregated_metrics and "accuracy" in aggregated_metrics:
            accuracy = aggregated_metrics["accuracy"]
            # The accuracy metric is a Scalar, which can be float, int, or bool.
            # Gauge.set expects a float.
            if isinstance(accuracy, (float, int)):
                # Set the accuracy gauge to the aggregated accuracy
                accuracy_gauge.set(accuracy)
                logging.info(f"Round {server_round}: Global accuracy set to {accuracy:.4f}")

        return aggregated_loss, aggregated_metrics


# --- ServerApp Factory Function ---
def server_fn(context: Context):

    """Construct components that set the ServerApp behaviour."""

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    

    # Initialize model parameters
    ndarrays = get_weights(Net())
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

# Start the Prometheus metrics server on port 8000
# The metrics will be available at http://localhost:8000

# Create and run the Flower ServerApp
app = ServerApp(server_fn=server_fn)

@app.lifespan()
def lifespan(context: Context):
    """Lifespan event to start Prometheus metrics server."""
    # Start the Prometheus HTTP server on port 8001
    start_http_server(8001)
    logging.info("Prometheus metrics server started on port 8001")
    yield
    logging.info("Shutting down Prometheus metrics server")