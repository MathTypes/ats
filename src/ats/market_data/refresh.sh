#!/bin/bash
# Example: sh -x src/ats/market_data/refresh.sh 2023-07-01
base_dir=$(dirname "$0")
rel_base_dir=$(realpath "$base_dir")
root_dir=$base_dir/../../..
echo $root_dir
NOW=$(date +"%Y-%m-%d")
TMPDIR=$root_dir/download/$NOW
mkdir -p $TMPDIR/FUT

wget -O $TMPDIR/FUT/data.zip "https://firstratedata.com/api/data_file?type=futures&period=full&adjustment=contin_UNadj&timeframe=5min&userid=fg1LcNsv8kWWMJIt0caCFQ"
cd $TMPDIR/FUT
unzip data.zip
cd $rel_base_dir/../../..

sh -x $base_dir/write_30min.sh $TMPDIR $1 $NOW
sh -x $base_dir/write_daily.sh $TMPDIR $1 $NOW
sh -x $base_dir/write_weekly.sh $TMPDIR $1 $NOW
sh -x $base_dir/write_monthly.sh $TMPDIR $1 $NOW
