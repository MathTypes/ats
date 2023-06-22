import logging
import sys


def init_logging():
    logging.basicConfig(
        format="%(asctime)s,%(msecs)03d %(levelname)-8s %(name)s[%(module)s:%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=logging.INFO,
    )

    root = logging.getLogger()
    root.setLevel(logging.ERROR)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    formatter = logging.Formatter(FORMAT)
    ch.setFormatter(formatter)
    root.addHandler(ch)
