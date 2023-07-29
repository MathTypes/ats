#!/bin/bash
# Example: sh -x src/ats/market_data/write_month.sh 2023-05-01 $PWD/data/firstratedata/full
base_dir=$(dirname "$0")
root_dir=$base_dir/../../..
echo $root_dir
tickers=$(cat "$base_dir/firstratedata_fut.txt")
export RAY_DATA_STRICT_MODE=0
for ticker in $tickers
do
    poetry run python3 src/ats/market_data/write_monthly_ts.py --ticker=$ticker --asset_type=FUT --start_date=$1 \
	   --input_dir=$2 --output_dir=$root_dir/data --freq=30min
done
