#!/bin/bash

# Get file size
filepath="$@"
new_filepath=`sed  s/ //g`
echo "mv '$filepath' $new_filepath"