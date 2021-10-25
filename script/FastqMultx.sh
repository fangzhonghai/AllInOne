#!/bin/bash
# 允许1错配;单barcode;barcode在reads2
fq1=$1
fq2=$2
barcode=$3
out_dir=$4
chip=$5
lane=$6
fastq-multx -B $barcode -m 1 -e $fq2 $fq1 -o $out_dir/${chip}_${lane}_%_2.fq.gz -o $out_dir/${chip}_${lane}_%_1.fq.gz
