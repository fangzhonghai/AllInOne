#!/usr/bin/env bash
workdir=$1
pipeline=$2

export PATH=$pipeline/tools:$PATH
Workdir=$workdir
fq_Q20_Q30=$pipeline/depthQC/fq_Q20_Q30.py
merge_QC=$pipeline/depthQC/merge_QC.py

ls $workdir/*/filter/*.fastp.json > $workdir/fastp.json.list
python3 $fq_Q20_Q30 $workdir/fastp.json.list $workdir/Q20_Q30.txt
if [ $? -ne 0 ]; then
     echo "Q20_Q30 extract error" >> $workdir/log
     exit 1
fi

ls $workdir/*/coverage/*.stat > $workdir/qc.list
python3 $merge_QC $workdir/Q20_Q30.txt $workdir/qc.list $workdir/QC.xlsx
if [ $? -ne 0 ]; then
     echo "QC merge error" >> $workdir/log
     exit 1
fi

touch $workdir/shell/FinalQC.sh.complete
echo `date` Done
echo `date` finalQC Done >> $workdir/log
