import argparse
import os
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

# The root directory where all partitioned datasets will be saved
DATASET_DIRECTORY = "datasets"


def save_dataset_to_disk(dataset_name: str, num_partitions: int):
    """
    Downloads the specified dataset, generates N IID partitions,
    and saves them to disk in an organized folder structure.

    Args:
        dataset_name (str): The name of the dataset to use.
                            Choices: "fashion_mnist", "mnist", "cifar10".
        num_partitions (int): The number of partitions to create.
    """
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

    # Define the specific output directory based on dataset name and number of partitions
    output_path = os.path.join(DATASET_DIRECTORY, dataset_name, f"{num_partitions}_partitions")
    
    # Create the nested directory structure if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    print(f"Partitions will be saved in: {output_path}")


    # Configure the partitioner
    partitioner = IidPartitioner(num_partitions=num_partitions)

    # Download and partition the dataset
    fds = FederatedDataset(
        dataset=hf_dataset_id,
        partitioners={"train": partitioner},
    )

    # Save each partition to disk inside the new directory structure
    for partition_id in range(num_partitions):
        partition = fds.load_partition(partition_id)
        # We split each partition into train/test sets
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

        # Define a unique path for each partition within the new structure
        file_path = os.path.join(output_path, f"{dataset_name}_part_{partition_id + 1}")
        partition_train_test.save_to_disk(file_path)
        print(f"Successfully saved partition {partition_id + 1} to: {file_path}")


if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(
        description="Download and save dataset partitions to disk."
    )

    # Add a required argument for the dataset name
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["fashion_mnist", "mnist", "cifar10"],
        help="The dataset to download (e.g., 'fashion_mnist', 'mnist', 'cifar10').",
    )

    # Add an optional argument for the number of partitions
    parser.add_argument(
        "--num-partitions",
        type=int,
        default=2,
        help="The number of partitions to create (default: 2).",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the provided arguments
    save_dataset_to_disk(dataset_name=args.dataset, num_partitions=args.num_partitions)
