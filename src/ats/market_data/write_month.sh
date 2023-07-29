#!/bin/bash
base_dir=$(dirname "$0")
echo $base_dir
tickers=$(cat "$base_dir/firstratedata_fut.txt")
for ticker in $tickers
do
    poetry run python3 src/ats/model/write_monthly_ts.py --ticker=$ticker --asset_type=FUT --start_date=2023-07-01 \
	   --end_date=2023-07-28 --input_dir=data/firstratedata/full --output_dir=data/firstratedata --freq=30min \
	   --start_date=2023-01-01
done
