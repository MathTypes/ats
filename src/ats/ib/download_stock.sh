#!/bin/bash

base_dir="/Volumes/Seagate Portable Drive/data"
done_file="$base_dir/$1.done"
if test -f "$done_file"; then
    echo "$FILE exists."
else
    python3 ib/download_hist.py $1  -p=4001 --start_date="20100101"  --end_date="20230422" --size="1 min"  --exchange=SMART -t TRADES --base-directory="$base_dir/trades" --security-type=STK  --debug --localsymbol=$1
    touch "$done_file"
fi

