#!/bin/bash
# This script constructs the command for the Flower SuperNode from environment variables.

# Exit immediately if a command exits with a non-zero status.
set -e

# Set default values for environment variables if they are not provided.
# This allows you to run the container without setting every single variable.
: "${SUPERLINK_IP:=localhost}"
: "${SUPERLINK_PORT:=9092}"
: "${DATASET_PATH:=/data/fashionmnist_part_1}"

# Assemble the command arguments into an array for robustness
COMMAND=(
    "flower-supernode"
    "--insecure"
    "--superlink=${SUPERLINK_IP}:${SUPERLINK_PORT}"
    "--node-config=client-n=${CLIENT_N}"  # This can be adjusted based on your client number
)

# If there are additional arguments passed to `docker run` or `docker-compose`,
# append them to the command.
if [ "$#" -gt 0 ]; then
    COMMAND+=("$@")
fi

# Announce the command that is about to be executed
echo "Executing command: ${COMMAND[*]}"

# Execute the final command, replacing the script process
exec "${COMMAND[@]}"
# Note: The `exec` command replaces the current shell with the command,