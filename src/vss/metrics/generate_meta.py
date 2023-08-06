#
# Usage: poetry run python3 src/vss/metrics/generate_meta.py --input=image.txt --output=image_meta.txt
#
import logging
from dataclasses import asdict
from filelock import FileLock
import os

import pandas as pd

from ats.util import config_utils
from ats.util import logging_utils

if __name__ == "__main__":
    parser = config_utils.get_arg_parser("Generate metadata")
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)

    args = parser.parse_args()
    config_utils.set_args(args)
    logging_utils.init_logging()

    file_df = pd.read_csv(args.input, header=None, names=["filepath"])
    new_df = pd.DataFrame(columns=["idx","file","class","label"])
    symbol_map = {}
    for idx, row in file_df.iterrows():
        filepath = row["filepath"]
        parsed_path = filepath.split("_")
        symbol = parsed_path[1]
        date_time = parsed_path[2]
        y_max = round(float(parsed_path[5])*100)
        y_min = round(float(parsed_path[6])*100)
        yhat_max = round(float(parsed_path[7])*100)
        yhat_min = round(float(parsed_path[8].replace(".png",""))*100)
        label_class = f"{y_max}_{y_min}_{yhat_max}_{yhat_min}"
        label = 100000 + y_max * 1000 + y_min * 100 + yhat_max * 10 + yhat_min
        if not symbol in symbol_map:
            symbol_map[symbol] = len(symbol_map) + 1
        symbol_label = symbol_map[symbol]
        row = {"idx":idx, "file":filepath, "class":label_class, "label":label}
        new_df.loc[new_df.shape[0]] = row
    new_df.to_csv(args.output)

        
    
