#!/usr/bin/bash

cd `dirname $0`

for f in ./*.pdf; do
    pdfcrop $f $f
done
