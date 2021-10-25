#!/usr/bin/env bash
workdir=$1
pipeline=$2
sampleID=$3

source $pipeline/anaconda3/bin/activate $pipeline/anaconda3/envs/biotools
export PATH=$pipeline/tools:$PATH
Workdir=$workdir/$sampleID/bwa

echo `date` Start sortSam
mkdir -p $workdir/javatmp
time gatk SortSam \
     -I $Workdir/$sampleID.raw.bam \
     -O $Workdir/$sampleID.sort.bam \
     -SO coordinate --TMP_DIR $workdir/javatmp --CREATE_INDEX
if [ $? -ne 0 ]; then
     echo "$sampleID sortSam error" >> $workdir/log
     exit 1
fi

touch $workdir/$sampleID/shell/SortSam.sh.complete
echo `date` Done
echo `date` $sampleID sortSam Done >> $workdir/log
