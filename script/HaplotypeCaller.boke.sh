#!/usr/bin/env bash
workdir=$1
pipeline=$2
sampleID=$3

source $pipeline/anaconda3/bin/activate $pipeline/anaconda3/envs/biotools
export PATH=$pipeline/tools:$PATH
Workdir=$workdir/$sampleID/gatk
hg38=$pipeline/hg38/hg38.fa
flank_bed=$pipeline/etc/PGxProble_CaptureRegion_boke_v1.sort_merge.flank200.bed

echo `date` Start HaplotypeCaller
mkdir -p $Workdir
time gatk HaplotypeCaller \
     --tmp-dir $workdir/javatmp \
     -R $hg38 \
     -L $flank_bed \
     -I $workdir/$sampleID/bwa/$sampleID.final.bam \
     -O $Workdir/$sampleID.gvcf.gz \
     -ERC GVCF
if [ $? -ne 0 ]; then
     echo "$sampleID HaplotypeCaller GVCF error" >> $workdir/log
     exit 1
fi

time gatk GenotypeGVCFs \
     --tmp-dir $workdir/javatmp \
     -R $hg38 \
     -L $flank_bed \
     -V $Workdir/$sampleID.gvcf.gz \
     -O $Workdir/$sampleID.final.vcf.gz
if [ $? -ne 0 ]; then
     echo "$sampleID GenotypeGVCFs error" >> $workdir/log
     exit 1
fi

touch $workdir/$sampleID/shell/HaplotypeCaller.boke.sh.complete
echo `date` Done
echo `date` $sampleID HaplotypeCaller Done >> $workdir/log
