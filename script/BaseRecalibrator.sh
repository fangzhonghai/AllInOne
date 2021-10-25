#!/usr/bin/env bash
workdir=$1
pipeline=$2
sampleID=$3

source $pipeline/anaconda3/bin/activate $pipeline/anaconda3/envs/biotools
export PATH=$pipeline/tools:$PATH
Workdir=$workdir/$sampleID/bwa
dbSNP=$pipeline/hg38/dbsnp_146.hg38.vcf.gz
GoldIndels=$pipeline/hg38/Mills_and_1000G_gold_standard.indels.hg38.vcf.gz
hg38=$pipeline/hg38/hg38.fa
flank_bed=$pipeline/etc/axk.7746-probes.add_SMN1.sort_merge.flank200.bed

echo `date` Start BQSR
time gatk BaseRecalibrator \
     --tmp-dir $workdir/javatmp \
     -I $Workdir/$sampleID.dup.bam \
     -O $Workdir/$sampleID.recal_data.grp \
     --known-sites $dbSNP \
     --known-sites $GoldIndels \
     -R $hg38 \
     -L $flank_bed
if [ $? -ne 0 ]; then
     echo "$sampleID BQSR error" >> $workdir/log
     exit 1
fi

echo `date` Start ApplyBQSR 
time gatk ApplyBQSR \
     --tmp-dir $workdir/javatmp \
     -R $hg38 \
     -I $Workdir/$sampleID.dup.bam \
     -O $Workdir/$sampleID.final.bam \
     -bqsr $Workdir/$sampleID.recal_data.grp 
if [ $? -ne 0 ]; then
     echo "$sampleID ApplyBQSR error" >> $workdir/log
     exit 1
fi

touch $workdir/$sampleID/shell/BaseRecalibrator.sh.complete
echo `date` Done
echo `date` $sampleID BaseRecalibrator Done >> $workdir/log
