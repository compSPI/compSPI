#!/usr/bin/env bash

set -e
set -o pipefail

source /opt/anaconda/etc/profile.d/conda.sh
conda activate base

echo "PATH: $PATH"

ls -ltrh /

echo "Executing command: "

bash -c "set -e; set -o pipefail; $1"