#!/bin/bash

command="${@:1}"
echo "Command: $command"

until $command
do 
    echo "Got exit code $?, try command again: $command"
    sleep 1
done
echo "Done"

