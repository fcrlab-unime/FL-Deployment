# Use the official Python 3.13 slim image as a base
FROM python:3.13-slim

# Install system-level dependencies
# Using --no-install-recommends to keep the image size smaller
# Cleaning up apt cache with rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    zlib1g-dev \
    libssl-dev \
    libsqlite3-dev \
    libreadline-dev \
    libbz2-dev \
    git \
    libffi-dev \
    liblzma-dev \
    libsndfile1 \
    htop \
    tmux \
    nano \
    gcc \
    pkg-config \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create mount points for the dataset and model volumes
RUN mkdir -p /app/data /app/model

# Copy and set permissions for the entrypoint script
COPY client_app.py /app/client_app.py
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Expose the default SuperNode port for clients (8080) and the SuperLink port (9092)
EXPOSE 8080
EXPOSE 9092

# Set the entrypoint to our custom script
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
