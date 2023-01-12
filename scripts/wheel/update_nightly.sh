#!/bin/bash

# stop immediately if a command exits with a non-zero status.
set -e

# download and extract the latest commit hidet
wget https://github.com/hidet-org/hidet/archive/refs/heads/main.zip -O hidet.zip
rm -rf hidet-main
unzip hidet.zip
cd hidet-main

# build the manylinux1 wheel
VERSION=`python3 scripts/wheel/current_version.py --nightly`
bash scripts/wheel/build_wheel_manylinux1.sh $VERSION

# copy the wheel to the deploy directory
if [ $# -eq 1 ]
then
  DEPLOY_DIR=$1
else
  DEPLOY_DIR=/var/www/html/whl/hidet
fi
WHEEL_PATH=`ls scripts/wheel/built_wheel/*.whl`
echo "Copying $WHEEL_PATH to $DEPLOY_DIR"
cp scripts/wheel/built_wheel/*.whl $DEPLOY_DIR

# remove the old wheels before 7 days ago in DEPLOY_DIR
find $DEPLOY_DIR -type f -mtime +7 -name '*.whl' -execdir echo -- 'Removing old wheel {}' \;
find $DEPLOY_DIR -type f -mtime +7 -name '*.whl' -execdir rm -- '{}' \;

# clean up
cd ..
rm -f hidet.zip
rm -rf hidet-main
