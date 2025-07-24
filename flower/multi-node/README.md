# Flower Multi-Node Federated Learning Deployment

This directory contains a multi-node Flower federated learning setup designed for distributed deployment across different physical machines or network nodes. Unlike the single-node setup, this configuration allows clients to run on separate machines while connecting to a centralized aggregator server.

## 🏗️ Architecture Overview

The multi-node deployment consists of two main components:

### Server Node (`/server/`)
- **Aggregator**: Flower SuperLink server coordinating federated learning
- **Monitoring Stack**:
  - **Prometheus**: Metrics collection and storage
  - **Grafana**: Visualization dashboard

### Client Nodes (`/client/`)
- **Federated Learning Clients**: Distributed across different machines
- **cAdvisor**: Local container resource monitoring
- **Dataset Storage**: Local dataset partitions

## 📁 Directory Structure

```
multi-node/
├── server/                        # Server node configuration
│   ├── .env                      # Server environment variables
│   ├── docker-compose.yml        # Server services
│   ├── Dockerfile                # Server container build
│   ├── server_app.py             # Flower server application
│   ├── client_app.py             # Client app reference
│   ├── entrypoint.sh             # Server startup script
│   ├── pyproject.toml            # Server dependencies
│   ├── requirements.txt          # Python requirements
│   ├── run.sh                    # Server startup script
│   └── config/                   # Monitoring configuration
│       ├── grafana.ini
│       ├── prometheus.yml
│       └── provisioning/
└── client/                       # Client node configuration
    ├── .env                      # Client environment variables
    ├── docker-compose.yml        # Client services
    ├── Dockerfile                # Client container build
    ├── entrypoint.sh             # Client startup script
    ├── requirements.txt          # Client requirements
    └── dataset/                  # Local dataset storage
```

## ⚙️ Configuration

### Main Configuration Files

The multi-node setup uses **separate configuration files** for server and client nodes:

#### Server Configuration (`server/.env`)
```bash
DATASET_NAME=cifar10
NUM_PARTITIONS=5
CLIENT_N=2
MODEL=resnet18
```

#### Client Configuration (`client/.env`)
```bash
SUPERLINK_IP=<SERVER_IP_ADDRESS>  # IP address of the server node
SUPERLINK_PORT=9092
CLIENT_N=2                 # Unique client identifier (MUST be manually set on each node)
```

⚠️ **MANDATORY**: Each physical client node must have its own `.env` file with a unique `CLIENT_N` value (1, 2, 3, etc.)

#### Server Dependencies (`server/pyproject.toml`)
Contains the complete project configuration including dependencies and Flower app components.

⚠️ **Critical Configuration Requirements**:
1. **Dataset Consistency**: `DATASET_NAME` and `NUM_PARTITIONS` in `server/.env` must match the available dataset partitions
2. **Network Configuration**: `SUPERLINK_IP` in `client/.env` must point to the actual IP address of the server node
3. **Client Identification**: Each client node must have a unique `CLIENT_N` value **manually configured** in its local `.env` file
4. **Port Accessibility**: Server ports (8080, 9092, 9093, 9094, 9095) must be accessible from client nodes

🔴 **MANDATORY**: Every physical client node requires manual configuration of `CLIENT_N` in its local `client/.env` file with unique values (1, 2, 3, 4, 5, etc.)

## � Dataset Management

The root directory contains two utility scripts for managing dataset partitions across multiple nodes:

### 1. Generate Dataset Partitions (`../../generate_dataset.py`)

This script downloads datasets and creates IID partitions for federated learning:

```bash
# Generate CIFAR-10 dataset with 5 partitions
python generate_dataset.py --dataset cifar10 --num-partitions 5

# Generate Fashion-MNIST with 2 partitions  
python generate_dataset.py --dataset fashion_mnist --num-partitions 2

# Generate MNIST with 3 partitions
python generate_dataset.py --dataset mnist --num-partitions 3
```

**Features**:
- Supports datasets: `cifar10`, `fashion_mnist`, `mnist`
- Creates IID (Independent and Identically Distributed) partitions
- Automatically organizes partitions in `datasets/<dataset_name>/<num_partitions>_partitions/`
- Splits each partition into train/test sets (80/20 split)

**Output Structure**:
```
datasets/
├── cifar10/
│   └── 5_partitions/
│       ├── cifar10_part_1/
│       ├── cifar10_part_2/
│       ├── cifar10_part_3/
│       ├── cifar10_part_4/
│       └── cifar10_part_5/
```

### 2. Distribute Partitions to Clients (`../../copy_partitions.py`)

This script automatically copies dataset partitions to remote client nodes via SSH/SCP:

```bash
# Distribute CIFAR-10 partitions to 5 clients
python copy_partitions.py --dataset cifar10 --num-partitions 5

# Distribute Fashion-MNIST to 2 clients
python copy_partitions.py --dataset fashion_mnist --num-partitions 2
```

**Configuration Required**:
Edit the script to configure your client network settings:

```python
# Client IP addresses mapping
CLIENT_IPS = {
    1: "172.x.x.1",   # client1 IP
    2: "172.x.x.2",   # client2 IP  
    3: "172.x.x.3",   # client3 IP
    4: "172.x.x.4",  # client4 IP
    5: "172.x.x.5",  # client5 IP
}

# SSH username for remote clients
REMOTE_USER = "remouser"

# Destination path on remote clients
REMOTE_BASE_DEST_PATH = "/home/remoteuser/FL-Deployment/flower/multi-node/client/dataset"
```

**Features**:
- Automatically creates remote directories on client nodes
- Uses SSH/SCP for secure file transfer
- Maps partitions to specific client IPs
- Handles error checking and validation

**Prerequisites**:
- SSH access to all client nodes
- SSH key-based authentication (recommended)
- Client nodes must be accessible from the server/management node

### Complete Dataset Workflow

1. **Generate partitions** (run from root directory):
   ```bash
   python generate_dataset.py --dataset cifar10 --num-partitions 5
   ```

2. **Configure client IPs** in `copy_partitions.py`

3. **Distribute partitions** to clients:
   ```bash
   python copy_partitions.py --dataset cifar10 --num-partitions 5
   ```

4. **Update configuration files** to match dataset settings:
   - `server/.env`: Set `DATASET_NAME=cifar10` and `NUM_PARTITIONS=5`
   - **Each `client/.env`**: **Manually set** appropriate `CLIENT_N` values on each physical node:
     - Node 1: `CLIENT_N=1`
     - Node 2: `CLIENT_N=2` 
     - Node 3: `CLIENT_N=3`
     - And so on...

## 🚀 Deployment Guide

### Prerequisites

- Docker and Docker Compose installed on all nodes
- Network connectivity between client and server nodes
- SSH access between server and client nodes (for dataset distribution)
- Models available on server node
- Dataset partitions generated and distributed (see Dataset Management section)

### Step 1: Configure Network Settings

1. **Determine Server IP Address**:
   ```bash
   # On server node, find the IP address
   hostname -I
   # or
   ip addr show
   ```

2. **Update Client Configuration**:
   Edit `client/.env` on each client node **individually**:
   ```bash
   SUPERLINK_IP=<SERVER_IP_ADDRESS>  # Replace with actual server IP
   SUPERLINK_PORT=9092
   CLIENT_N=<UNIQUE_CLIENT_ID>       # MANDATORY: 1, 2, 3, etc. (unique per node)
   ```

   🔴 **CRITICAL**: Each physical client node must have its `.env` file manually updated with a unique `CLIENT_N` value:
   - Client Node 1: `CLIENT_N=1`
   - Client Node 2: `CLIENT_N=2`
   - Client Node 3: `CLIENT_N=3`
   - And so on...

### Step 2: Prepare Datasets

#### Option A: Automated Distribution (Recommended)

Use the provided scripts for automatic dataset generation and distribution:

```bash
# 1. Generate dataset partitions (from root directory)
python generate_dataset.py --dataset cifar10 --num-partitions 5

# 2. Configure client IPs in copy_partitions.py
# Edit CLIENT_IPS, REMOTE_USER, and REMOTE_BASE_DEST_PATH

# 3. Distribute partitions to all clients
python copy_partitions.py --dataset cifar10 --num-partitions 5
```

#### Option B: Manual Distribution

If you prefer manual distribution, ensure the correct dataset partition is available on each client node:

```bash
# Example for client 1 with CIFAR-10
# Copy the appropriate dataset partition to client/dataset/
scp -r datasets/cifar10/5_partitions/cifar10_part_1 user@client1:/path/to/client/dataset/
```

⚠️ **Important**: Ensure the dataset structure matches the expected paths in the client configuration.

### Step 3: Deploy Server Node

On the server machine:

```bash
cd server/
docker compose up --build -d
```

This will start:
- Aggregator (Flower SuperLink) on ports 8080, 9092, 9093
- Prometheus on port 9095  
- Grafana on port 9094

### Step 4: Deploy Client Nodes

On each client machine **after manually configuring the `.env` file**:

⚠️ **Before deployment**: Ensure each client's `.env` file has the correct `CLIENT_N` value:
```bash
# Verify the configuration on each node
cat client/.env
# Should show: CLIENT_N=1 (or 2, 3, 4, etc. - unique per node)
```

```bash
cd client/
docker compose up --build -d
```

This will start:
- Flower SuperNode (client)
- cAdvisor for monitoring

### Step 5: Execute Federated Learning

Once all nodes are running, start the federated learning session on the **server node**:

```bash
docker exec aggregator /bin/bash -c "flwr run . embedded-federation"
```

## 📊 Monitoring and Logs

### Server Monitoring

Access monitoring dashboards on the server node:

- **Grafana Dashboard**: http://`<SERVER_IP>`:9094
  - Default credentials: admin/admin
  - Federated learning metrics and system monitoring

- **Prometheus**: http://`<SERVER_IP>`:9095
  - Raw metrics and queries

- **Flower Dashboard**: http://`<SERVER_IP>`:9093
  - Federated learning session monitoring

### Container Logs

Monitor federated learning progress:

```bash
# On server node
docker logs -f aggregator

# On client nodes  
docker logs -f client1  # (or client2, client3, etc.)
```

### Client Monitoring

Each client node provides local monitoring:
- **cAdvisor**: http://`<CLIENT_IP>`:8080
  - Container resource usage
  - Local system metrics

## 🔧 Customization

### Adding More Client Nodes

1. **Set up new client machine**:
   - Install Docker and Docker Compose
   - Copy the `client/` directory
   - Ensure dataset partition is available

2. **Configure client**:
   ```bash
   # Edit client/.env MANUALLY on each physical node
   SUPERLINK_IP=<SERVER_IP>
   SUPERLINK_PORT=9092
   CLIENT_N=<NEW_UNIQUE_ID>  # e.g., 3, 4, 5, etc. - MUST be unique per node
   ```

   🔴 **MANDATORY**: The `CLIENT_N` value must be manually set and unique on each physical client node.

3. **Update server configuration**:
   ```bash
   # Edit server/.env to reflect total partitions
   NUM_PARTITIONS=<NEW_TOTAL_COUNT>
   ```

4. **Deploy client**:
   ```bash
   cd client/
   docker compose up --build -d
   ```

### Changing Datasets

1. **Update server configuration** (`server/.env`):
   ```bash
   DATASET_NAME=fashion_mnist
   NUM_PARTITIONS=5
   ```

2. **Update server dependencies** (`server/pyproject.toml`) to match dataset requirements

3. **Distribute new dataset partitions** to client nodes:
   ```bash
   # On each client node
   cp -r ../../datasets/fashion_mnist/5_partitions/fashion_mnist_part_<N> client/dataset/
   ```

4. **Restart all nodes**:
   ```bash
   # Server node
   cd server/ && docker compose down && docker compose up --build -d
   
   # Each client node
   cd client/ && docker compose down && docker compose up --build -d
   ```

### Network Security

For production deployments, consider:
- Using TLS/SSL certificates (remove `--insecure` flag)
- Configuring firewall rules
- Using VPN or private networks
- Implementing authentication mechanisms

## 🛠️ Troubleshooting

### Common Issues

1. **Connection Refused**:
   - Verify server IP address in `client/.env`
   - Check firewall settings on server node
   - Ensure server ports are accessible

2. **Dataset Not Found**:
   - Verify dataset partitions exist on client nodes: `ls -la client/dataset/`
   - Check dataset paths and naming conventions match configuration
   - Ensure dataset configuration matches between server and clients
   - Re-run dataset distribution: `python copy_partitions.py --dataset <name> --num-partitions <n>`
   - Verify SSH access for automatic distribution

3. **Client Registration Failed**:
   - Verify unique `CLIENT_N` values for each client - **each physical node must have manually configured unique values**
   - Check network connectivity between nodes
   - Review server logs for connection attempts
   - Ensure each client's `.env` file has been manually updated with correct `CLIENT_N`

4. **Port Conflicts**:
   - Ensure required ports are available on all nodes
   - Check for conflicts with existing services

5. **Configuration Mismatch**:
   - Verify `DATASET_NAME` and `NUM_PARTITIONS` consistency
   - Ensure all clients point to correct server IP
   - Check that client count matches expected partitions

6. **Dataset Distribution Issues**:
   - Verify SSH key-based authentication is set up
   - Check client IP addresses in `copy_partitions.py`
   - Ensure remote paths exist and are writable
   - Test SSH connectivity: `ssh user@client_ip "echo 'SSH works'"`

### Network Diagnostics

```bash
# Test connectivity from client to server
ping <SERVER_IP>
telnet <SERVER_IP> 9092

# Check port availability on server
netstat -tlnp | grep :9092

# Verify container status
docker compose ps

# Check container logs
docker logs aggregator
docker logs client1
```

### Useful Commands

```bash
# Server node commands
cd server/
docker compose ps                    # Check service status
docker compose logs aggregator       # View server logs
docker compose restart aggregator    # Restart server

# Client node commands  
cd client/
docker compose ps                    # Check client status
docker compose logs client          # View client logs
docker compose restart client       # Restart client

# Complete cleanup (all nodes)
docker compose down -v              # Stop and remove volumes
docker system prune -f              # Clean up Docker resources

# Dataset management commands (from root directory)
python generate_dataset.py --help   # Show dataset generation options
python copy_partitions.py --help    # Show distribution options

# Verify dataset distribution
ssh user@client1 "ls -la /path/to/client/dataset/"
ssh user@client2 "ls -la /path/to/client/dataset/"

# Test SSH connectivity to all clients
for ip in 172.x.x.1 172.x.x.2; do ssh user@$ip "echo 'Client $ip: OK'"; done
```

## 📋 Dependencies

### Server Dependencies
- **Flower**: `>=1.15.2` - Federated learning framework
- **PyTorch**: `2.6.0` - Deep learning framework
- **TorchVision**: `0.21.0` - Computer vision utilities
- **Flower Datasets**: `>=0.5.0` - Dataset utilities

### Client Dependencies
- **Docker & Docker Compose** - Container orchestration
- **Network connectivity** - Access to server node
- **Local storage** - Dataset partition storage

## 📄 License

Apache-2.0

## 🤝 Contributing

When modifying the multi-node setup:
1. Test network connectivity between nodes
2. Verify monitoring dashboards work correctly  
3. Ensure logs are accessible on both server and client nodes
4. Test with multiple client configurations
5. Update this README with any configuration changes
