#!/usr/bin/env bash
workdir=$1
pipeline=$2
sampleID=$3

export PATH=$pipeline/tools:$PATH
export PYTHONPATH=$pipeline/Cyrius:$PYTHONPATH
Workdir=$workdir/$sampleID/cyp2d6

echo `date` Start CYP2D6 Genotyping
aldy genotype -p wxs -g CYP2D6 -o $Workdir/$sampleID.cyp2d6.aldy.wxs.tsv $workdir/$sampleID/bwa/$sampleID.final.bam
if [ $? -ne 0 ]; then
     echo "$sampleID aldy error" >> $workdir/log
     exit 1
fi

ls $workdir/$sampleID/bwa/$sampleID.final.bam > $Workdir/bam.list
python3 $pipeline/Cyrius/star_caller.py \
     --outDir $Workdir --genome 38 --threads 2 \
     --prefix $sampleID.cyp2d6.Cyrius --manifest $Workdir/bam.list
if [ $? -ne 0 ]; then
     echo "$sampleID Cyrius error" >> $workdir/log
     exit 1
fi

touch $workdir/$sampleID/shell/CYP2D6.sh.complete
echo `date` Done
echo `date` $sampleID CYP2D6 Genotyping Done >> $workdir/log
