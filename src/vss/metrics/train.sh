#!/bin/bash
base_dir=$(dirname "$0")
root_dir=$base_dir/../../..
root_dir=$(realpath $root_dir)
echo $root_dir
export $(grep -v '^#' $base_dir/../.env | xargs)
export $(grep -v '^#' $base_dir/../.env-local-no-docker | xargs)
poetry run python3 src/vss/metrics/train.py --config-path=$root_dir/conf --config-name=build_search
