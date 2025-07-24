# Federated Learning Deployment Comparison

This repository provides a comprehensive comparison of three federated learning deployment tools through practical implementations and benchmarks. The project aims to evaluate different approaches to federated learning deployment across various environments and use cases.

## 🎯 Project Overview

This repository compares three federated learning frameworks:

1. **Flower** - Popular open-source federated learning framework (✅ **COMPLETE**)
2. **OpenFL** - Intel's federated learning framework (🚧 **PLANNED**)
3. **Fleet** - Our Kubernetes-native federated learning proposal (🚧 **PLANNED**)

## 📊 Evaluation Components

### Datasets (3 supported)
- **CIFAR-10**: 32x32 color images, 10 classes
- **Fashion-MNIST**: 28x28 grayscale fashion items, 10 classes  
- **MNIST**: 28x28 grayscale handwritten digits, 10 classes

### Models (3 implemented)
- **CNN**: Convolutional Neural Network (for Fashion-MNIST)
- **DNN**: Deep Neural Network (for MNIST)
- **ResNet-18**: Residual Network (for CIFAR-10)

### Deployment Scenarios
- **Single-Node**: All components on one machine
- **Multi-Node**: Distributed across multiple physical machines
- **Kubernetes**: Cloud-native deployment (Fleet proposal)

## 🏗️ Repository Structure

```
FL-Deployment/
├── flower/                          # Flower framework implementations
│   ├── single-node/                # Single-node setup (✅ COMPLETE)
│   │   ├── README.md               # Detailed setup instructions
│   │   ├── docker-compose.yml      # Main services
│   │   ├── docker-compose.clients.yml
│   │   └── ...
│   └── multi-node/                 # Multi-node setup (✅ COMPLETE)
│       ├── README.md               # Detailed setup instructions  
│       ├── server/                 # Server node configuration
│       ├── client/                 # Client node configuration
│       └── ...
├── openfl/                         # OpenFL implementations (🚧 PLANNED)
├── fleet/                          # Fleet K8s implementation (🚧 PLANNED)
├── models/                         # Shared model implementations
│   ├── cnn.py                     # CNN for Fashion-MNIST
│   ├── dnn.py                     # DNN for MNIST
│   └── resnet18.py                # ResNet-18 for CIFAR-10
├── datasets/                       # Generated dataset partitions
│   ├── cifar10/
│   ├── fashion_mnist/
│   └── mnist/
├── generate_dataset.py             # Dataset partitioning utility
├── copy_partitions.py             # Multi-node dataset distribution
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Generate Dataset Partitions

Create federated learning datasets with IID partitions:

```bash
# Generate CIFAR-10 with 5 partitions
python generate_dataset.py --dataset cifar10 --num-partitions 5

# Generate Fashion-MNIST with 2 partitions
python generate_dataset.py --dataset fashion_mnist --num-partitions 2

# Generate MNIST with 3 partitions  
python generate_dataset.py --dataset mnist --num-partitions 3
```

### 2. Choose Your Deployment

#### Flower Framework (Available Now)

**Single-Node Setup**: All components on one machine
```bash
cd flower/single-node/
# See flower/single-node/README.md for detailed instructions
```

**Multi-Node Setup**: Distributed across multiple machines
```bash
cd flower/multi-node/
# See flower/multi-node/README.md for detailed instructions
```

#### OpenFL Framework (Coming Soon)
```bash
cd openfl/
# Implementation planned
```

#### Fleet K8s Framework (Coming Soon)
```bash
cd fleet/
# Kubernetes-native implementation planned
```

## 📋 Roadmap

### ✅ Completed Features

- **Flower Single-Node Deployment**
  - Docker Compose orchestration
  - 2-client default setup with scaling support
  - Integrated monitoring (Prometheus, Grafana, cAdvisor)
  - All 3 datasets and models supported
  - Automated client generation

- **Flower Multi-Node Deployment**
  - Distributed deployment across physical machines
  - SSH-based dataset distribution
  - Network configuration management
  - Individual client monitoring
  - All 3 datasets and models supported

- **Dataset Management**
  - IID partitioning for all datasets
  - Automated distribution utilities
  - Configurable partition counts

- **Model Library**
  - CNN, DNN, and ResNet-18 implementations
  - Self-contained training and testing logic
  - Dataset-specific optimizations

### 🚧 In Progress

- **Performance Benchmarking**
  - Metrics collection standardization
  - Comparative analysis framework
  - Resource usage profiling

### 📅 Planned Features

#### Phase 2: OpenFL Implementation
- [ ] OpenFL single-node setup
- [ ] OpenFL multi-node setup  
- [ ] OpenFL dataset integration
- [ ] OpenFL model compatibility
- [ ] Performance comparison with Flower

#### Phase 3: Fleet K8s Implementation
- [ ] Kubernetes operator development
- [ ] Helm charts for deployment
- [ ] Auto-scaling capabilities
- [ ] Cloud provider integration
- [ ] Service mesh integration

#### Phase 4: Advanced Features
- [ ] Non-IID data partitioning
- [ ] Differential privacy implementations
- [ ] Advanced aggregation algorithms
- [ ] Cross-framework benchmarking
- [ ] Performance optimization guides

## 🔧 Utility Scripts

### Dataset Generation (`generate_dataset.py`)

Creates IID partitions for federated learning:

```bash
python generate_dataset.py --dataset <dataset_name> --num-partitions <count>
```

**Supported datasets**: `cifar10`, `fashion_mnist`, `mnist`

**Features**:
- Downloads datasets automatically
- Creates train/test splits (80/20)
- Organizes in structured directories
- Supports any number of partitions

### Dataset Distribution (`copy_partitions.py`)

Distributes dataset partitions to remote nodes via SSH:

```bash
python copy_partitions.py --dataset <dataset_name> --num-partitions <count>
```

**Features**:
- SSH/SCP-based secure transfer
- Automatic remote directory creation
- IP address mapping configuration
- Error handling and validation

**Configuration**: Edit script variables for your network setup:
```python
CLIENT_IPS = {1: "IP1", 2: "IP2", ...}
REMOTE_USER = "username"
REMOTE_BASE_DEST_PATH = "/path/to/client/dataset"
```

## 📚 Adding New Components

### Adding New Models

1. Create model file in `models/` directory:
```python
# models/your_model.py
import torch.nn as nn

class YourModel(nn.Module):
    def __init__(self):
        # Model definition
    
    def load_data(self, dataset_path):
        # Data loading logic
    
    def train(self, parameters, config):
        # Training logic
    
    def test(self, parameters, config):
        # Testing logic
```

2. Update framework configurations to use your model
3. Test with existing datasets

### Adding New Datasets

1. Add dataset support to `generate_dataset.py`:
```python
dataset_mapping = {
    "your_dataset": "huggingface/dataset-name",
    # ... existing datasets
}
```

2. Create corresponding model if needed
3. Update documentation

### Adding New Frameworks

1. Create framework directory: `your_framework/`
2. Implement single-node and multi-node setups
3. Integrate with existing models and datasets
4. Add comprehensive README
5. Update this root README

## 📖 Detailed Documentation

### Flower Framework
- **Single-Node Setup**: See [flower/single-node/README.md](flower/single-node/README.md)
  - Complete setup instructions
  - Docker Compose configuration
  - Monitoring and troubleshooting
  - Customization options

- **Multi-Node Setup**: See [flower/multi-node/README.md](flower/multi-node/README.md)
  - Distributed deployment guide
  - Network configuration
  - SSH setup and dataset distribution
  - Client management

### OpenFL Framework
- Documentation coming soon

### Fleet K8s Framework  
- Documentation coming soon

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-framework`
3. Follow existing patterns for consistency
4. Add comprehensive documentation
5. Test with all supported datasets and models
6. Submit pull request

### Development Guidelines

- Use Docker for containerization
- Provide both single-node and multi-node setups
- Include monitoring and logging
- Write detailed README files
- Test with all 3 datasets and models
- Follow security best practices

## 📄 License

Apache-2.0

## 📞 Support

For framework-specific issues:
- **Flower**: Check `flower/single-node/README.md` or `flower/multi-node/README.md`
- **General**: Open an issue in this repository

## 🏆 Goals

This project aims to provide:
1. **Practical comparison** of federated learning frameworks
2. **Production-ready** deployment examples
3. **Comprehensive benchmarking** data
4. **Best practices** for federated learning deployment
5. **Educational resources** for the community

Join us in advancing federated learning deployment practices!
