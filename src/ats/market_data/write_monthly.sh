#!/bin/bash
# Example: sh -x src/ats/market_data/write_month.sh 2023-05-01 $PWD/data/firstratedata/full
base_dir=$(dirname "$0")
root_dir=$base_dir/../../..
echo $root_dir
tickers=$(cat "$base_dir/firstratedata_fut.txt")
export RAY_DATA_STRICT_MODE=0
input_dir=$1
start_date=$2
end_date=$3
for ticker in $tickers
do
    poetry run python3 src/ats/market_data/write_monthly_ts.py --ticker=$ticker \
	   --asset_type=FUT --start_date=$start_date --end_date=$end_date \
	   --input_dir=$input_dir --output_dir=$root_dir/data --freq=1MS
done
