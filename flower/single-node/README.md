# Flower Single-Node Federated Learning Deployment

This directory contains a complete single-node Flower federated learning setup with monitoring capabilities. The setup uses Docker Compose to orchestrate multiple containers including the aggregator server, multiple clients, and monitoring infrastructure.

## 🏗️ Architecture Overview

The deployment consists of:
- **Aggregator**: Flower SuperLink server coordinating federated learning
- **Clients**: Multiple federated learning clients (default: 2 clients)
- **Monitoring Stack**:
  - **Prometheus**: Metrics collection and storage
  - **Grafana**: Visualization dashboard
  - **cAdvisor**: Container resource monitoring

## 📁 Directory Structure

```
single-node/
├── docker-compose.yml          # Main services (aggregator, monitoring)
├── docker-compose.clients.yml  # Client containers configuration
├── Dockerfile.server          # Server container build
├── Dockerfile.client          # Client container build
├── server_app.py              # Flower server application
├── client_app.py              # Flower client application
├── server_entrypoint.sh       # Server startup script
├── client_entrypoint.sh       # Client startup script
├── pyproject.toml             # Python dependencies
├── requirements.txt           # Python requirements
├── generate_client_compose.sh # Script to generate client configs
├── run.sh                     # Convenience startup script
└── config/                    # Monitoring configuration
    ├── grafana.ini           # Grafana configuration
    ├── prometheus.yml        # Prometheus configuration
    └── provisioning/         # Grafana dashboards and datasources
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Datasets prepared in `../../datasets/` directory
- Models available in `../../models/` directory

### 1. Start All Services

Build and start all containers (aggregator, clients, and monitoring):

```bash
docker compose -f docker-compose.yml -f docker-compose.clients.yml up --build -d
```

This command will:
- Build the server and client images
- Start the aggregator (Flower SuperLink)
- Launch monitoring services (Prometheus, Grafana, cAdvisor)
- Start federated learning clients

### 2. Execute Federated Learning

Once all containers are running, start the federated learning session:

```bash
docker exec aggregator /bin/bash -c "flwr run . embedded-federation"
```

This executes the federated learning workflow using Flower's embedded federation mode.

## 📊 Monitoring and Logs

### Container Logs

Monitor the federated learning process by viewing container logs:

```bash
# Monitor aggregator logs
docker logs -f aggregator

# Monitor client logs
docker logs -f client1
docker logs -f client2

# Monitor all containers
docker logs -f aggregator & docker logs -f client1 & docker logs -f client2
```

### Grafana Dashboard

Access the Grafana monitoring dashboard:
- **URL**: http://localhost:9094
- **Default credentials**: admin/admin (configure on first login)
- **Features**:
  - Container resource usage
  - System metrics
  - Custom federated learning metrics

### Prometheus Metrics

Access raw metrics via Prometheus:
- **URL**: http://localhost:9095
- **Features**:
  - Query metrics directly
  - Explore available metrics
  - Debug monitoring setup

### cAdvisor

Container resource monitoring:
- **URL**: http://localhost:8080
- **Features**:
  - Real-time container stats
  - Resource usage graphs
  - Container health monitoring

## ⚙️ Configuration

### Main Configuration Files

There are **two key configuration files** that must be synchronized for the deployment to work correctly:

#### 1. `.env` File
Contains environment variables that control the deployment:
```bash
export DATASET_NAME=cifar10
export NUM_PARTITIONS=2
export SUPERLINK_IP=aggregator
export SUPERLINK_PORT=9092
```

#### 2. `pyproject.toml`
Contains Python dependencies and project configuration. **Important**: The dataset configuration here must match the `.env` file.

⚠️ **Critical**: The `DATASET_NAME` and `NUM_PARTITIONS` values must be identical in both files for the system to work correctly.

### Client Configuration Generation

The `generate_client_compose.sh` script reads the `.env` file and automatically generates the `docker-compose.clients.yml` file with the correct number of clients and dataset paths:

```bash
# Generate client configuration based on .env file
./generate_client_compose.sh
```

This script will:
- Read `DATASET_NAME` and `NUM_PARTITIONS` from `.env`
- Generate client services (client1, client2, etc.)
- Configure correct dataset volume mounts
- Set up proper environment variables for each client

### Current Default Configuration

With the default `.env` settings:
- **Dataset**: CIFAR-10
- **Partitions**: 2 clients
- **Dataset Paths**: 
  - `client1`: `../../datasets/cifar10/2_partitions/cifar10_part_1`
  - `client2`: `../../datasets/cifar10/2_partitions/cifar10_part_2`
- **SuperLink**: `aggregator:9092`

### Model Sharing

All containers share the models directory: `../../models`

## 🔧 Customization

### Changing Dataset or Number of Partitions

1. **Update `.env` file**:
   ```bash
   export DATASET_NAME=fashion_mnist  # Change dataset
   export NUM_PARTITIONS=5            # Change number of clients
   export SUPERLINK_IP=aggregator
   export SUPERLINK_PORT=9092
   ```

2. **Update `pyproject.toml`** to match the dataset configuration

3. **Regenerate client configuration**:
   ```bash
   ./generate_client_compose.sh
   ```

4. **Restart the deployment**:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.clients.yml down
   docker compose -f docker-compose.yml -f docker-compose.clients.yml up --build -d
   ```

### Adding More Clients

1. Update `NUM_PARTITIONS` in `.env` file
2. Ensure corresponding dataset partitions exist in `../../datasets/`
3. Run the generation script:
   ```bash
   ./generate_client_compose.sh
   ```

### Manual Client Configuration

Alternatively, you can manually edit `docker-compose.clients.yml`, but using the generation script ensures consistency with the `.env` configuration.

### Modifying Training Parameters

Edit the `pyproject.toml` file to adjust:
- Dependencies
- Model configurations  
- Training parameters

## 🛠️ Troubleshooting

### Common Issues

1. **Port Conflicts**: Ensure ports 8080, 9092, 9094, 9095 are available
2. **Dataset Not Found**: Verify dataset paths exist in `../../datasets/`
3. **Model Loading Issues**: Check model files in `../../models/`
4. **Configuration Mismatch**: Ensure `DATASET_NAME` and `NUM_PARTITIONS` are identical in both `.env` and `pyproject.toml`
5. **Client Generation Issues**: Run `./generate_client_compose.sh` after updating `.env`

### Useful Commands

```bash
# Check container status
docker compose ps

# Restart specific service
docker compose restart aggregator

# View resource usage
docker stats

# Access container shell
docker exec -it aggregator /bin/bash

# Stop all services
docker compose -f docker-compose.yml -f docker-compose.clients.yml down

# Clean up everything (including volumes)
docker compose -f docker-compose.yml -f docker-compose.clients.yml down -v
```

## 📋 Dependencies

- **Flower**: `>=1.15.2` - Federated learning framework
- **PyTorch**: `2.6.0` - Deep learning framework
- **TorchVision**: `0.21.0` - Computer vision utilities
- **Flower Datasets**: `>=0.5.0` - Dataset utilities

## 📄 License

Apache-2.0

## 🤝 Contributing

When modifying the setup:
1. Test with the provided quick start commands
2. Verify monitoring dashboards work correctly
3. Ensure logs are accessible and meaningful
4. Update this README with any configuration changes
