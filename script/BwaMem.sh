#!/usr/bin/env bash
workdir=$1
pipeline=$2
sampleID=$3

export PATH=$pipeline/tools:$PATH
Workdir=$workdir/$sampleID/bwa
hg38=$pipeline/hg38/hg38.fa

echo `date` Start bwaMem
mkdir -p $Workdir
time bwa mem \
     -K 1000000 -t 8 -M -a \
     -R "@RG\tID:FAMILY_all\tSM:$sampleID\tLB:$sampleID\tPL:COMPLETE" \
     $hg38 \
     $workdir/$sampleID/filter/$sampleID.filter_1.fq.gz \
     $workdir/$sampleID/filter/$sampleID.filter_2.fq.gz \
     | samtools view -S -b -o $Workdir/$sampleID.raw.bam -
if [ $? -ne 0 ]; then
     echo "$sampleID bwaMem error" >> $workdir/log
     exit 1
fi

touch $workdir/$sampleID/shell/BwaMem.sh.complete
echo `date` Done
echo `date` $sampleID bwaMem Done >> $workdir/log
