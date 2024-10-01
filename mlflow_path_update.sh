#!/bin/bash

# Set the directory where mlruns is located
MLRUNS_DIR="mlruns"

# Find all meta.yaml files and update the paths
find "$MLRUNS_DIR" -name "meta.yaml" -type f -print0 | xargs -0 sed -i -E '
    s#(source: )file:///.*/mlruns/#\1/workspace/mlruns/#g
    s#(storage_location: )file:///.*/mlruns/#\1/workspace/mlruns/#g
'

echo "Path updates complete."