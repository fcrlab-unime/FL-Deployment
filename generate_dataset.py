import argparse
import os
import random
import numpy as np
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

# The root directory where all partitioned datasets will be saved
DATASET_DIRECTORY = "datasets"


def set_seeds(seed: int):
    """
    Sets the random seeds for Python and NumPy to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    print(f"✅ Random seeds set to: {seed}")


def save_dataset_to_disk(dataset_name: str, num_partitions: int, seed: int):
    """
    Downloads the specified dataset, generates N IID partitions reproducibly,
    and saves them to disk in an organized folder structure.

    Args:
        dataset_name (str): The name of the dataset to use.
        num_partitions (int): The number of partitions to create.
        seed (int): The random seed for reproducibility.
    """
    # Set all random seeds for full reproducibility
    set_seeds(seed)

    # Map the user-friendly name to the official Hugging Face dataset identifier
    dataset_mapping = {
        "fashion_mnist": "zalando-datasets/fashion_mnist",
        "mnist": "mnist",
        "cifar10": "cifar10",
    }

    hf_dataset_id = dataset_mapping.get(dataset_name)
    if not hf_dataset_id:
        raise ValueError(
            f"Dataset '{dataset_name}' is not supported. "
            f"Choose from {list(dataset_mapping.keys())}"
        )

    print(f"Downloading and partitioning '{dataset_name}' into {num_partitions} partitions...")

    # Define a specific output directory that includes the seed
    output_path = os.path.join(
        DATASET_DIRECTORY, dataset_name, f"{num_partitions}_partitions_seed{seed}"
    )
    os.makedirs(output_path, exist_ok=True)
    print(f"Partitions will be saved in: {output_path}")

    # Configure the partitioner. It uses np.random internally, which is now seeded.
    partitioner = IidPartitioner(num_partitions=num_partitions)

    # Download and partition the dataset deterministically
    fds = FederatedDataset(
        dataset=hf_dataset_id,
        partitioners={"train": partitioner},
    )

    # Save each partition to disk
    for partition_id in range(num_partitions):
        partition = fds.load_partition(partition_id)
        # Use the global seed to make the train/test split reproducible
        partition_train_test = partition.train_test_split(test_size=0.2, seed=seed)

        # Define a unique path for each partition
        file_path = os.path.join(output_path, f"{dataset_name}_part_{partition_id + 1}")
        partition_train_test.save_to_disk(file_path)
        print(f"Successfully saved partition {partition_id + 1} to: {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and save dataset partitions to disk reproducibly."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["fashion_mnist", "mnist", "cifar10"],
        help="The dataset to download.",
    )

    parser.add_argument(
        "--num-partitions",
        type=int,
        default=2,
        help="The number of partitions to create (default: 2).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The random seed for reproducibility (default: 42).",
    )

    args = parser.parse_args()

    save_dataset_to_disk(
        dataset_name=args.dataset,
        num_partitions=args.num_partitions,
        seed=args.seed,
    )