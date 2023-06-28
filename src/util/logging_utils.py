import logging
from os import path
import sys
import logging.config


def init_logging():
    #log_file_path = path.join(path.dirname(path.abspath(__file__)), '../util/logging.conf')
    #logging.info(f"log_file_path:{log_file_path}")
    #logging.config.fileConfig(log_file_path)
    logging.basicConfig(
        format="%(asctime)s,%(msecs)03d %(levelname)-8s %(name)s[%(module)s:%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=logging.INFO,
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    #ch = logging.StreamHandler(sys.stdout)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    formatter = logging.Formatter(FORMAT)
    ch.setFormatter(formatter)
    root.addHandler(ch)
    logging.info(f"init")
    #exit(0)
