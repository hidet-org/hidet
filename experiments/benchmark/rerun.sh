#!/bin/bash

prog="python end2end.py"
 
$prog
while [ $? -ne 0 ]; do
    $prog
done

