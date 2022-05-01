#!/bin/bash

#echo "$@"
prog="$@"
echo "running $prog"

$prog
while [ $? -ne 0 ]; do
    $prog
    sleep 1
done
