#!/usr/bin/env bash
workdir=$1
pipeline=$2
sampleID=$3

source $pipeline/anaconda3/bin/activate $pipeline/anaconda3/envs/biotools
export PATH=$pipeline/tools:$PATH
Workdir=$workdir/$sampleID/bwa

echo `date` Start MarkDuplicates
time gatk MarkDuplicates \
     -I $Workdir/$sampleID.sort.bam \
     -O $Workdir/$sampleID.dup.bam \
     -M $Workdir/$sampleID.dup.metrics \
     --CREATE_INDEX --CLEAR_DT false --TMP_DIR $workdir/javatmp
if [ $? -ne 0 ]; then
     echo "$sampleID dupMark error" >> $workdir/log
     exit 1
fi

touch $workdir/$sampleID/shell/DupMark.sh.complete
echo `date` Done
echo `date` $sampleID dupMark Done >> $workdir/log
