#!/bin/bash
ls -ltr /data/trades/FUT/1_min/$1*/|grep csv|awk '{print $9}'|sort