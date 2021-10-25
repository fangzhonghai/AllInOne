#!/usr/bin/env bash
workdir=$1
pipeline=$2
sampleID=$3

export PATH=$pipeline/tools:$PATH
Workdir=$workdir/$sampleID/hla

echo `date` Start BamtoFastq
mkdir -p $Workdir
samtools view -h -b $workdir/$sampleID/bwa/$sampleID.final.bam chr6:28510120-33480577 > $Workdir/$sampleID.mhc.bam
if [ $? -ne 0 ]; then
     echo "$sampleID extract mhc bam error" >> $workdir/log
     exit 1
fi
samtools view -b -f 12 $workdir/$sampleID/bwa/$sampleID.final.bam > $Workdir/$sampleID.unmap.bam
if [ $? -ne 0 ]; then
     echo "$sampleID extract mhc unmap error" >> $workdir/log
     exit 1
fi
samtools merge -f $Workdir/$sampleID.hla.bam $Workdir/$sampleID.mhc.bam $Workdir/$sampleID.unmap.bam
if [ $? -ne 0 ]; then
     echo "$sampleID extract merge hla bam error" >> $workdir/log
     exit 1
fi
samtools sort -n $Workdir/$sampleID.hla.bam > $Workdir/$sampleID.hla.sort.bam
if [ $? -ne 0 ]; then
     echo "$sampleID sort hla bam error" >> $workdir/log
     exit 1
fi
bedtools bamtofastq -i $Workdir/$sampleID.hla.sort.bam -fq $Workdir/$sampleID.hla_1.fq -fq2 $Workdir/$sampleID.hla_2.fq
if [ $? -ne 0 ]; then
     echo "$sampleID bamtofastq error" >> $workdir/log
     exit 1
fi

touch $workdir/$sampleID/shell/BamtoFastq.sh.complete
echo `date` Done
echo `date` $sampleID BamtoFastq Done >> $workdir/log
