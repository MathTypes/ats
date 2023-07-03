import sys
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df = pd.read_parquet(sys.argv[1])

print(df)
print(df.info())
print(df.describe())
