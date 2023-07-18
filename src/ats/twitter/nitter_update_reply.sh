#!/bin/bash
maxsize=10

# Get file size
filepath="$@"
filesize=$(stat -c%s "$filepath")
difference=$(expr $filesize - $maxsize)
# The following doesn't work
if [ $difference -gt 0 ]
then
    PYTHONPATH=. python3 twitter/download_tweet_by_id.py --id_file=$filepath
fi
