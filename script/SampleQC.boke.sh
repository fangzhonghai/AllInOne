#!/usr/bin/env bash
workdir=$1
pipeline=$2
sampleID=$3

export PATH=$pipeline/tools:$PATH
Workdir=$workdir/$sampleID/coverage
dict=$pipeline/hg38/hg38.dict
depthStat=$pipeline/depthQC/depth.stat.pl
bed=$pipeline/etc/PGxProble_CaptureRegion_boke_v1.sort_merge.bed

echo `date` Start sampleQC
mkdir -p $Workdir
time perl $depthStat \
     -o $Workdir/$sampleID \
     -m $workdir/$sampleID/bwa/$sampleID.final.bam \
     -d $dict -b $bed -f 200
if [ $? -ne 0 ]; then
     echo "$sampleID sampleQC depthStat error" >> $workdir/log
     exit 1
fi

touch $workdir/$sampleID/shell/SampleQC.boke.sh.complete
echo `date` Done
echo `date` $sampleID sampleQC Done >> $workdir/log
