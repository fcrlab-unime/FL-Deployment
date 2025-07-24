#!/bin/bash

# Usage: ./generate-clients.sh <DATASET_NAME> <NUM_CLIENTS>
# Load environment variables from .env file if it exists

if [ -f .env ]; then
  echo "📦 Loading environment variables from .env"
  set -a              # Export all variables that follow
  source .env
  set +a
else
  echo "⚠️  .env file not found. Exiting."
  exit 1
fi

NUM_PARTITIONS=$NUM_PARTITIONS
DATASET_NAME=$DATASET_NAME
SUPERLINK_IP=$SUPERLINK_IP
SUPERLINK_PORT=$SUPERLINK_PORT

echo "Generating docker-compose.clients.yml for $NUM_PARTITIONS partitions of $DATASET_NAME dataset with SuperLink at $SUPERLINK_IP:$SUPERLINK_PORT"
# Start the override file
cat <<EOF > docker-compose.clients.yml

services:
EOF

for ((i=1; i<=NUM_PARTITIONS; i++)); do

  cat <<EOF >> docker-compose.clients.yml

  client$i:
    build:
      context: .
      dockerfile: Dockerfile.client
    container_name: client$i
    labels:
      - "io.cadvisor.scrape=true"
      - "io.cadvisor.container=true"
    command: []
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - ../../datasets/${DATASET_NAME}/${NUM_PARTITIONS}_partitions/${DATASET_NAME}_part_${i}:/app/dataset/${DATASET_NAME}/${NUM_PARTITIONS}_partitions/${DATASET_NAME}_part_${i}
      - ../../../models:/app/models
    environment:
      - SUPERLINK_IP=${SUPERLINK_IP}
      - SUPERLINK_PORT=${SUPERLINK_PORT}
      - container_name=client$i
      - DOCKER_HOST_IP=host.docker.internal
      - CLIENT_N=${i}
    stop_signal: SIGINT
    depends_on:
      - cadvisor
EOF
done

echo "✅ docker-compose.clients.yml generated with $NUM_CLIENTS clients."
