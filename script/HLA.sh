#!/usr/bin/env bash
workdir=$1
pipeline=$2
sampleID=$3

export PATH=$pipeline/tools:$PATH
export PATH=$pipeline/hlahd.1.3.0/bin:$PATH
Workdir=$workdir/$sampleID/hla

echo `date` Start HLA Genotyping
if [ -d $Workdir/$sampleID ]; then
     rm -rf $Workdir/$sampleID
fi
hlahd.sh -t 3 -m 90 -f $pipeline/hlahd.1.3.0/freq_data \
     $Workdir/$sampleID.hla_1.fq $Workdir/$sampleID.hla_2.fq \
     $pipeline/hlahd.1.3.0/HLA_gene.split.3.32.0.txt $pipeline/hlahd.1.3.0/dictionary \
     $sampleID $Workdir
if [ $? -ne 0 ]; then
     echo "$sampleID HLA Genotyping error" >> $workdir/log
     exit 1
fi

touch $workdir/$sampleID/shell/HLA.sh.complete
echo `date` Done
echo `date` $sampleID HLA Genotyping Done >> $workdir/log
