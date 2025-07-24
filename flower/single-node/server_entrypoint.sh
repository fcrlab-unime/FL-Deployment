#!/bin/sh
# entrypoint.sh

# Exit immediately if a command exits with a non-zero status.
set -e

# Execute the Flower SuperLink, pointing to the ServerApp object.
# The server will be accessible on port 8080.
exec flower-superlink --insecure 
