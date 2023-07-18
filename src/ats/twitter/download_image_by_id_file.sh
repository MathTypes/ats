#!/bin/bash
maxsize=400

# Get file size
filesize=$(stat -c%s "$1")
difference=$(expr $filesize - $maxsize)
# The following doesn't work
if [ $difference -gt 0 ]
then
     PYTHONPATH=. python3 twitter/download_tweet_image_by_reply.py --reply_file=$1
fi
