#!/bin/bash

cd `dirname $0`

for f in ./*.py; do
  echo $f
  python $f
done

bash ./pdfs/0_crop.sh
