#!/usr/bin/env bash
workdir=$1
pipeline=$2
sampleID=$3

export PATH=$pipeline/tools:$PATH
Workdir=$workdir/$sampleID/offtarget
flank=50
bed2flank=$pipeline/statistics/bed2flank.pl
bed=$pipeline/etc/PGxProble_CaptureRegion_roche_v2.sort_merge.bed
hg38=$pipeline/GRCh38/GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna

echo `date` Start OffTarget stat
mkdir -p $Workdir
cp $bed $Workdir/target.bed
if [ $flank -ne 0 ]; then
     perl $bed2flank $Workdir/target.bed $hg38 $flank $Workdir/target.$flank.bed
     bed=$Workdir/target.$flank.bed
fi

bedtools genomecov -ibam $workdir/$sampleID/bwa/$sampleID.final.bam -bg > $Workdir/$sampleID.bedgraph
if [ $? -ne 0 ]; then
     echo "$sampleID sampleQC genomecov error" >> $workdir/log
     exit 1
fi

bedtools subtract -a $Workdir/$sampleID.bedgraph -b $bed > $Workdir/$sampleID.bedgraph.offtarget
if [ $? -ne 0 ]; then
     echo "$sampleID sampleQC subtract error" >> $workdir/log
     exit 1
fi

touch $workdir/$sampleID/shell/OffTarget.roche.sh.complete
echo `date` Done
echo `date` $sampleID OffTarget Done >> $workdir/log