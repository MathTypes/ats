PYTHONPATH=.. python3 write_monthly_ts.py --ticker=$2 --asset_type=$1 --start_date=2008-01-01 --end_date=2023-05-01 --input_dir=. --output_dir=data --freq=30min
