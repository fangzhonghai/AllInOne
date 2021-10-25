#!/usr/bin/perl
use warnings;
use strict;
die "usage:perl $0 <input> <fa> <flank_bp> <output>\n",unless @ARGV==4;
open IN,$ARGV[0],or die $!;
unless (-f "$ARGV[1].fai"){
	`samtools faidx $ARGV[1]`;
}
open FA,"$ARGV[1].fai",or die $!;
my %len;
while (<FA>){
	chomp;
	my @array=split/\t/,$_;
	$len{$array[0]}=$array[1];
}
close FA;

my $flank=$ARGV[2],or die $!;
open OUT,">$ARGV[3].tmp",or die $!;
my %hash;
while (<IN>){
	next if (/start/i);
	my @array=split/\t/,$_;
	my $start=$array[1]-$flank;
	my $end=$array[2]+$flank;
	if ($array[1]-$flank<0){
		$start=0;
	}
	if ($array[2]+$flank>$len{$array[0]}){
		$end=$len{$array[0]};
	}
	print OUT "$array[0]\t$start\t$end\n";
#	$hash{$array[0]}{$start}{$end}=1;
}
close IN;
close OUT;
`bedtools merge -i $ARGV[3].tmp >$ARGV[3]`;
`rm -rf $ARGV[3].tmp`;
