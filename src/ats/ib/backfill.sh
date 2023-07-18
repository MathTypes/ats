#!/bin/bash
for dt in "20210301" "20210401" "20210501" "20210601" "20210701" "20210801" "20210901" "20211001" "20211101" "20211201" "20220101" "20220201" "20220301" "20220401" "20220501" "20220601" "20220701" "20220801" "20220901" "20221001" "20221101" "20221201" "20230101" "20230201" "20230301" "20230401"
do
    python3 ib/download_fut.py $1 --start_date=$dt --end_date=$dt --port=4001 --duration='1 D' --base-directory='/Volumes/Seagate Portable Drive/data/trades' --security-type=FUT --size='1 min' --data-type=TRADES --port=4001
done    