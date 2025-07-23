import argparse
import os
import subprocess
import sys

# --- Configuration ---
# Map the client number (from the partition name) to its IP address.
# Add all your client IPs here.
CLIENT_IPS = {
    1: "172.17.6.4",
    2: "172.17.6.6", 
    3: "172.17.6.9", 
    4: "172.17.6.13", 
    5: "172.17.6.15",
    # Add more clients as needed
}

# Your username on the remote client machines.
REMOTE_USER = "larosa"

# The absolute path on the remote machine where the dataset partitions will be copied.
# The script will place the partition folders (e.g., 'fashion_mnist_part_1') inside this directory.
REMOTE_BASE_DEST_PATH = "/home/larosa/FL-Deployment/flower/multi-node/client/dataset"
# --- End Configuration ---


def distribute_partitions(dataset_name: str, num_partitions: int):
    """
    Creates the remote directory and copies dataset partitions to clients.

    Args:
        dataset_name (str): The name of the dataset (e.g., 'fashion_mnist').
        num_partitions (int): The total number of partitions to distribute.
    """
    print(f"--- Starting distribution for dataset '{dataset_name}' ---")

    if num_partitions > len(CLIENT_IPS):
        print(f"Error: You requested to distribute {num_partitions} partitions, but only "
              f"{len(CLIENT_IPS)} client IPs are defined in the CLIENT_IPS dictionary.")
        sys.exit(1)

    for i in range(1, num_partitions + 1):
        client_id = i
        partition_name = f"{dataset_name}_part_{client_id}"
        
        local_source_path = os.path.join(
            "datasets", dataset_name, f"{num_partitions}_partitions", partition_name
        )

        if not os.path.isdir(local_source_path):
            print(f"\nERROR: Source directory not found: '{local_source_path}'")
            print("Please ensure you have generated the datasets first.")
            sys.exit(1)

        remote_ip = CLIENT_IPS.get(client_id)
        if not remote_ip:
            print(f"\nERROR: IP address for client {client_id} not found. Skipping.")
            continue

        print(f"\n>>> Processing Client {client_id} at {remote_ip}")

        # MODIFICATION START: Automatically create the remote directory
        # ---------------------------------------------------------------------
        # Define the full destination path on the remote machine.
        remote_dest_parent_path = os.path.join(
            REMOTE_BASE_DEST_PATH, dataset_name, f"{num_partitions}_partitions"
        )

        # Construct an SSH command to create the directory. 
        # 'mkdir -p' creates parent directories as needed and doesn't fail if it already exists.
        ssh_command = [
            "ssh",
            f"{REMOTE_USER}@{remote_ip}",
            f"mkdir -p {remote_dest_parent_path}"
        ]

        print(f"  1. Ensuring remote directory exists...")
        print(f"     > {' '.join(ssh_command)}")
        
        try:
            # Execute the ssh command to create the directory
            subprocess.run(
                ssh_command, check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"  > ERROR: Failed to create directory on client {client_id}.")
            print(f"  > STDERR: {e.stderr.strip()}")
            sys.exit(1)
        except FileNotFoundError:
            print("\nERROR: 'ssh' command not found. Is OpenSSH client installed?")
            sys.exit(1)
        # ---------------------------------------------------------------------
        # MODIFICATION END

        # Construct the scp command to copy the partition directory
        scp_command = [
            "scp",
            "-r",
            local_source_path,
            f"{REMOTE_USER}@{remote_ip}:{remote_dest_parent_path}/",
        ]

        print(f"  2. Copying partition data...")
        print(f"     > {' '.join(scp_command)}")

        try:
            subprocess.run(
                scp_command, check=True, capture_output=True, text=True
            )
            print("  > Success!")
        except subprocess.CalledProcessError as e:
            print(f"  > ERROR: Failed to copy to client {client_id}.")
            print(f"  > STDERR: {e.stderr.strip()}")
            sys.exit(1)
        except FileNotFoundError:
            print("\nERROR: 'scp' command not found. Is OpenSSH client installed?")
            sys.exit(1)

    print(f"\n--- Successfully distributed {num_partitions} partitions. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distribute dataset partitions to remote clients via SCP."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["fashion_mnist", "mnist", "cifar10"],
        help="The name of the dataset to distribute.",
    )
    parser.add_argument(
        "--num-partitions",
        type=int,
        required=True,
        help="The number of client partitions to copy.",
    )
    args = parser.parse_args()
    distribute_partitions(args.dataset, args.num_partitions)