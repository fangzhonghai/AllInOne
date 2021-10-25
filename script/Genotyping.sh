#!/usr/bin/env bash
workdir=$1
pipeline=$2
sampleID=$3

export PATH=$pipeline/tools:$PATH
Workdir=$workdir/$sampleID/genotyping
genotyping=$pipeline/genotyping/variant.py

echo `date` Start genotyping
mkdir -p $Workdir
time python3 $genotyping \
     -bam $workdir/$sampleID/bwa/$sampleID.final.bam \
     -vcf $workdir/$sampleID/gatk/$sampleID.final.vcf.gz \
     -out $Workdir/$sampleID.variant.tsv -drug variant
if [ $? -ne 0 ]; then
     echo "$sampleID variant genotyping error" >> $workdir/log
     exit 1
fi

time python3 $genotyping \
     -bam $workdir/$sampleID/bwa/$sampleID.final.bam \
     -vcf $workdir/$sampleID/gatk/$sampleID.final.vcf.gz \
     -out $Workdir/$sampleID.ace.tsv -drug ace
if [ $? -ne 0 ]; then
     echo "$sampleID ace genotyping error" >> $workdir/log
     exit 1
fi

touch $workdir/$sampleID/shell/Genotyping.sh.complete
echo `date` Done
echo `date` genotyping Done >> $workdir/log
