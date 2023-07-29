#!/bin/bash
base_dir=$(dirname "$0")
root_dir=$base_dir/../../..
echo $root_dir
tickers=$(cat "$base_dir/firstratedata_fut.txt")
for ticker in $tickers
do
    poetry run python3 src/ats/market_data/write_monthly_ts.py --ticker=$ticker --asset_type=FUT --start_date=$1 \
	   --input_dir=$2 --output_dir=$root_dir/data --freq=30min
done
