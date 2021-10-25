#!/usr/bin/env bash
workdir=$1
pipeline=$2
sampleID=$3
fq1=$4
fq2=$5

export PATH=$pipeline/tools:$PATH
Workdir=$workdir/$sampleID/filter

echo `date` Start cat fq
mkdir -p $Workdir
fq1_arry=(${fq1//,/ })
fq2_arry=(${fq1//,/ })
fq1_len=${#fq1_arry[@]}
fq2_len=${#fq2_arry[@]}
if [ $fq1_len -ne $fq2_len ]; then
     echo "$sampleID fq1 fq2 number unequal" >> $workdir/log
     exit 1
fi
if [ $fq1_len -ne 1 ];then
     cat ${fq1//,/ } > $Workdir/$sampleID.cat_1.fq.gz
     cat ${fq2//,/ } > $Workdir/$sampleID.cat_2.fq.gz
     fq1=$Workdir/$sampleID.cat_1.fq.gz
     fq2=$Workdir/$sampleID.cat_2.fq.gz
fi

echo `date` Start fastp
time fastp \
     -i $fq1 -I $fq2 \
     -o $Workdir/$sampleID.filter_1.fq.gz -O $Workdir/$sampleID.filter_2.fq.gz \
     -j $Workdir/$sampleID.fastp.json -h $Workdir/$sampleID.fastp.html --thread 6
if [ $? -ne 0 ]; then
     echo "$sampleID fastp error" >> $workdir/log
     exit 1
fi

touch $workdir/$sampleID/shell/Filter.sh.complete
echo `date` Done
echo `date` $sampleID filter Done >> $workdir/log
