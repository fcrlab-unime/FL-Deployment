# Federated Learning Server Setup 🤖

This document outlines the steps to configure, run, and monitor the central federated learning server (aggregator) and its associated monitoring stack using Docker.

---

## ⚙️ Configuration

Before launching, you can adjust the federated learning parameters.

#### Federated Learning Parameters

The primary configuration for the training process is located in the **`pyproject.toml`** file, under the `[tool.flwr.app.config]` section. You can modify key hyperparameters here:

* `num-server-rounds`: The total number of federated learning rounds.
* `fraction-evaluate`: The fraction of clients to use for evaluation.
* `local-epochs`: The number of epochs each client runs locally.
* `learning-rate`: The learning rate for the client-side optimizer.
* `batch-size`: The batch size for client-side training.

#### Monitoring Configuration

The configuration for Prometheus and Grafana is located in the `config/` directory. You typically don't need to change this for initial setup.

---

## ▶️ Running the Server

Running the server is a two-step process: first, you launch the services with Docker Compose, and second, you start the federation from within the aggregator container.

1.  **Launch Docker Services**
    From within this `server` directory, build and run the containers in detached mode:
    ```bash
    docker-compose up --build -d
    ```
    This will start three services: `aggregator`, `prometheus`, and `grafana`.

2.  **Start the Federation**
    After the containers are running (both clients and server), you must `exec` into the `aggregator` container and manually start the Flower federation process.
    ```bash
    docker exec -it aggregator /bin/bash -c "flwr run . embedded-federation"
    ```
    If there are `min_available_clients` (configurable in the **`server_app.py`** script) the federated learning session starts.

---

## 📊 Monitoring

The system includes a complete monitoring stack to observe the federation's progress and performance.

* **Flower Dashboard**: Access the live Flower SuperLink dashboard to see connected clients and server status at **[http://localhost:9093](http://localhost:9093)**.

* **Prometheus**: View the raw metrics collected from the aggregator at **[http://localhost:9095](http://localhost:9095)**.

* **Grafana**: Visualize the metrics on pre-built dashboards at **[http://localhost:9094](http://localhost:9094)**. Use `admin` for both the username and password on your first login.

**Important**: The custom metrics server that exposes **training/validation loss and accuracy** is part of the `server_app`. It will only begin exposing metrics after the `flwr run` command is executed and the federation starts. The metrics will no longer be available after the session finishes.