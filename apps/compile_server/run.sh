#!/bin/bash

docker build -t compile_server .
docker run -d -p 3281:3281 --rm compile_server
