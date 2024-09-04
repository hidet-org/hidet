# Nightly built wheels server

This folder contains the scripts that are used to serve the downloading the nightly-built wheels.


The two scripts in this folder
- `add-crobtab-record.sh`: add a crontab record to the system. The crontab record will let the system launch the `update-nightly.sh` script at 0:00 every day.
- `update-nightly.sh`: this script will pull the latest commit from the main branch of `hidet-org/hidet` repo and build a wheel with versions like `0.4.1.dev20240721` and put it to the `whl/hidet` subdirectory of our wheel server for the users downloading.

Setup steps:
1. Launch a web server. 
2. Run the `add-crontab-record.sh`.
