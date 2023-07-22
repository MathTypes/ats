from collections import Counter
import linecache
import logging
import os
import tracemalloc
from time import sleep


def display_top(snapshot, key_type="lineno", limit=30):
    snapshot = snapshot.filter_traces(
        (
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        )
    )
    top_stats = snapshot.statistics(key_type)

    logging.info(f"Top {limit} lines")
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        logging.info(f"#{index}: {filename}{frame.lineno} {stat.size/1024} KiB")
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            logging.info(f"    {line}")

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        logging.info(f"{len(other)} other: {size/1024} KiB")
    total = sum(stat.size for stat in top_stats)
    logging.info(f"Total allocated size: {total/1024} KiB")


def count_prefixes():
    sleep(2)  # Start up time.
    counts = Counter()
    fname = "/usr/share/dict/american-english"
    with open(fname) as words:
        words = list(words)
        for word in words:
            prefix = word[:13]
            counts[prefix] += 1
            sleep(0.0001)
    most_common = counts.most_common(3)
    sleep(3)  # Shut down time.
    return most_common


def start_trace_malloc():
    tracemalloc.start()


def take_snapshot():
    most_common = count_prefixes()
    logging.info(f"Top prefixes:{most_common}")

    snapshot = tracemalloc.take_snapshot()
    display_top(snapshot)
