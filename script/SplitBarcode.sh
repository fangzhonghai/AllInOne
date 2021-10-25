#!/bin/bash
# BGI软件splitBarcode2.0.0;允许1错配;单barcode;barcode在reads2
fq1=$1
fq2=$2
barcode=$3
out_dir=$4
#chip=$5
#lane=$6

# glibc-2.14
glibc214=/share/app/glibc-2.14
export PATH=$glibc214/bin:$PATH
export CPATH=$glibc214/include:$CPATH
export LIBRARY_PATH=$glibc214/lib64:$glibc214/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$glibc214/lib64:$glibc214/lib:$LD_LIBRARY_PATH

splitBarcode -B $barcode -1 $fq1 -2 $fq2 -o $out_dir -b 200 10 1 -r
