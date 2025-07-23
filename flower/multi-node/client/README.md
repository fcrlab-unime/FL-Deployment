# 🚀 Federated Learning Client Setup

This guide outlines the steps required to configure and run a **federated learning client node** using Docker.

---

## ✅ Prerequisites

Before launching a client, ensure the following are set up:

### 1. Environment Variables

Create a `.env` file in the `client` directory **or** export the variables in your shell. These are used by `docker-compose.yml`.

**Required variables:**

* `CLIENT_N`: A unique number for each client (e.g., `1`, `2`, `3`). This identifies the client and its dataset partition.
* `SUPERLINK_IP`: The **IP address of the machine running the server**.

**Example `.env` file:**

```env
CLIENT_N=1
SUPERLINK_IP=192.168.1.100
```

---

### 2. Dataset Partition 💾

Place the client's data partition in the `dataset/` folder.

The data loading logic in `model/model.py` expects the path defined by the `DATASET_PATH` environment variable (configured in `docker-compose.yml`).

**Default pattern:**

```
/app/dataset/fashionmnist_part_${CLIENT_N}
```

So for `CLIENT_N=1`, ensure the dataset folder is:

```
dataset/fashionmnist_part_1/
```

You can use the `generate_dataset.py` script in the project root to partition your datasets accordingly.

---

### 3. Model Implementation 🧠

You must implement your model and training logic in `model/model.py`.

The following functions are required:

* `Net()`: Defines the model architecture (e.g., using PyTorch or TensorFlow).
* `load_data_from_disk(partition_id)`: Loads the specific data partition from disk.
* `get_weights(model)`: Serializes and returns the model’s weights.
* `set_weights(model, weights)`: Loads the given weights into the model.
* `train(...)`: Implements local training logic.
* `test(...)`: Evaluates the model on local test data.

> **Note:** If you modify function signatures or core logic in `client_app.py`, rebuild the Docker image.

---

## ▶️ Running the Client

1. Ensure your `.env` file is created or export the environment variables manually.
2. From the `client` directory, run the following:

```bash
# Example for starting client 1
export CLIENT_N=1
export SUPERLINK_IP=<YOUR_SERVER_IP>
docker-compose up --build
```
3. (optional) Check logs

```bash
docker logs -f client1
```
Repeat this on each client machine, ensuring that `CLIENT_N` is **unique** for each one.

---

