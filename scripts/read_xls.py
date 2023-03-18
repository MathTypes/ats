
import argparse
import pandas as pd

parser = argparse.ArgumentParser(
    description='A script to read xls file'
)
parser.add_argument("--file", type=str, required=True)

args = parser.parse_args()

# read by default 1st sheet of an excel file
dataframe1 = pd.read_excel(args.file)

print(dataframe1)
