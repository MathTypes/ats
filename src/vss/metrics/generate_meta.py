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
        symbol = parsed_path[0]
        date_time = parsed_path[1]
        if not symbol in symbol_map:
            symbol_map[symbol] = len(symbol_map) + 1
        symbol_label = symbol_map[symbol]
        new_df = new_df.append({"idx":idx, "file":filepath, "class":symbol, "label":symbol_label}, ignore_index=True)
    new_df.to_csv(args.output)

        
    
