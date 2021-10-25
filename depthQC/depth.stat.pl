use strict;
use Getopt::Long;
use Math::SigFigs;
use FindBin qw($Bin);
=head1 NAME:

	Program:	depth.pl
	Version:	0.1.3
	Author:		Wang Yaoshen (wangyaoshen@genomics.cn) Gao Changxin (gaochangxin@genmics.cn)
	Program Date:	2015-11-18
			v1.0	2013-8-26
			v1.1	2013-12-27
			v1.2	2014-2-24
			v1.3	2014-3-14
			v2.0	2014-4-6
	Description:	depth statistics

=head1 USAGE:

	perl depth.pl [options]
	-o	STR	output file prefix [./]
	-m	FILE	sam/bam file
	-d	FILE	reference dict file
	-g	STR	gender (M/F) [M]
	-w	INT	window size [1000000]
	-l	INT	max read length [150]
	-b	FILE	target bed file, can be not used if for WGS
	-f	INT	flank length, must be used with -b, can be not used if don't statist the flank region 

=cut

my ($output,$bam,$dict,$gender,$win,$len,$bed,$flank);
GetOptions(
	"o:s" => \$output,
	"m=s" => \$bam,
	"d=s" => \$dict,
	"g=s" => \$gender,
	"w=i" => \$win,
	"l=i" => \$len,
	"b:s" => \$bed,
	"f:i" => \$flank
);
die `pod2text $0` unless($bam and $dict);
$output||="./";
$output.="." unless $output=~/\/$/;
$gender||="M";
$win||=1000000;
$len||=150;

#get the chr length from the dict file
#my $whole_genome=0;	#the total length of whole genome
my %chr_len;
my %chr_ord;
$chr_ord{"chr"}=0;
open(DT,$dict) || die "$!";
#@SQ	SN:chr1	LN:249250621	UR:file:	M5:
my $ord=1;
while(<DT>){
	chomp;
	next if /^\@HD/;
	my @t=split;
	$t[1]=~s/^SN://;
	$t[2]=~s/^LN://;
	next if $gender eq "F" and $t[1] eq "chrY";
	#$whole_genome+=$t[2];
	$chr_len{$t[1]}=$t[2];
	$chr_ord{$t[1]}=$ord;
	$ord++;
}
close DT;
my %chr_chr=reverse(%chr_ord);

my $targetsize;
my @window;	#window array
my @window_rmdup;	#window array
my $offset=0;	#offset to the position
my $map_reads=0;	#the number of mapped reads
my $count_GC=0;
my $count_AT=0;
my $count_N=0;
my $total_reads=0;	#the number of total reads
my $read_depth=0;	#reads total depth to the whole genome
my $read_depth_rmdup=0;#####################################################################################
my $pair_map_reads=0;	#the number of pair mapped reads
my @insert_size;	#the number of every insert size
my $indel_reads=0;  #the number of indel reads
my $mismatch_reads=0;	#the number of reads mismatch mapped
my $duplication=0;

my $target_map_reads=0;	#the number of reads mapped to the target region
my $target_map_dup=0; ####################################################################################
my $target_read_depth=0;	#reads total depth to the target region
my $target_read_depth_rmdup=0;	#reads total depth to the target region
#my $target_uniq_map_reads=0;	#the number of unique mapped reads to the target region

my %chr_depth;
my %chr_cover;
my %chr_size;
my @target_depth;	#the number of every depth to the target region
my @target_depth_rmdup;	#the number of every depth to the target region
my @target_cumu_depth;	#the cumulative number of every depth to the target region
my @target_cumu_depth_rmdup;	#the cumulative number of every depth to the target region
my $flank_map_reads=0;	#the number of reads mapped to the flank region
my $flank_read_depth=0;	#reads total depth to the flank region
my $flank_read_depth_rmdup=0;	#reads total depth to the flank region
my @flank_depth;	#the number of every depth to the flank region
my @flank_depth_rmdup;	#the number of every depth to the flank region
my @flank_cumu_depth;	#the cumulative number of every depth to the flank region
my @flank_cumu_depth_rmdup;	#the cumulative number of every depth to the flank region

my @region=();	#region position (target/flank,chr,start-1,end)
if($bed){
	#get the target region
	open(BED,$bed) || die "$!";
	while(<BED>){
		chomp;
		my @t=(split)[0,1,2];
		$targetsize += $t[2]-$t[1];
		next unless exists $chr_ord{$t[0]};
		unshift(@t,"target");
		push(@region,\@t);
	}
	close BED;
	@region=sort {$chr_ord{$a->[1]} <=> $chr_ord{$b->[1]} or $a->[3] <=> $b->[3] or $a->[2] <=> $b->[2]} @region;	#target region sort

	#remove the overlap and get the flank region
	for(my $i=@region-1; $i>=0; $i--){
		if($i==@region-1){	#the last flank region
			if($flank){
				my @rf=("flank",$region[-1][1],$region[-1][3],$region[-1][3]+$flank);
				$rf[3]=$chr_len{$region[-1][1]} if $rf[3]>$chr_len{$region[-1][1]};
				push(@region,\@rf);
			}
		}else{
			if($region[$i][1] ne $region[$i+1][1]){	#the near region in the different chr
				if($flank){
					my @lf=("flank",$region[$i+1][1],$region[$i+1][2]-$flank,$region[$i+1][2]);
					$lf[2]=0 if $lf[2]<0;
					my @rf=("flank",$region[$i][1],$region[$i][3],$region[$i][3]+$flank);
					$rf[3]=$chr_len{$region[$i][1]} if $rf[3]>$chr_len{$region[$i][1]};
					splice(@region,$i+1,0,(\@rf,\@lf));	#insert the flank region
				}
			}else{	#the near region in the same chr
				if($region[$i+1][2]<=$region[$i][3]){	#overlap
					$region[$i+1][2]=$region[$i][2] if $region[$i+1][2]>$region[$i][2];	#combine the region
					splice(@region,$i,1);	#delete this region
				}else{
					if($flank){
						if($region[$i+1][2]-$region[$i][3]<=$flank){	#flank region overlap
							my @f=("flank",$region[$i][1],$region[$i][3],$region[$i+1][2]);
							splice(@region,$i+1,0,(\@f));	#insert the flank region
						}else{
							my @lf=("flank",$region[$i+1][1],$region[$i+1][2]-$flank,$region[$i+1][2]);
							my @rf=("flank",$region[$i][1],$region[$i][3],$region[$i][3]+$flank);
							splice(@region,$i+1,0,(\@rf,\@lf));	#insert the flank region
						}
					}
				}
			}
		}
	}
	if($flank){	#the first flank region
		my @lf=("flank",$region[0][1],$region[0][2]-$flank,$region[0][2]);
		$lf[2]=0 if $lf[2]<0;
		unshift(@region,\@lf);
	}
}

if($bam=~/.bam$/){
	open(IN,"samtools view -h $bam|") || die "$!";
}else{
	open(IN,$bam) || die "$!";
}
my $chr="chr";
my @left_region=();
my @left_region_rmdup=();
my $r=0;	#array @region index of reads number statistics
my $d=0;	#array @region index of depth statistics
my $max_len=0;
while(<IN>){
#FCC11YMACXX:6:2116:16079:66414#NCGNAGTA	163	chr1	14620	12	62M28S	=	14683	153	CAGCTTGTCCTGGCTGTGTCCATGTCAGAGCAATGGCCCAAGTCTGGGTCTGGGGGGAAAGGGGTCATGCAGCCCCCTACGATTCCCAGT	=;.AA?F>CACE8B?E>E=@>@?;=@A?A:C>A@ECDBAABC5CCAD31@CCAC;@6?A<A>..&1?=B+/:@A>::??A=<@#######	BD:Z:LLPPQNOMPNNPONOOLLMPNOPQMPOPOOPPMNQOOOJOMNPPNPOKOPNPOKKKKNNDNPKKOPOPQPPQQPKKKOORQRROOPLLLL MD:Z:33C23G4 PG:Z:MarkDuplicates RG:Z:FAMILY_DLW XG:i:0 BI:Z:OORRQNOOQOQQPNQQOOOQOQRRPRRRQQRQOPQPNOLQNOQQPQPLQQPQPLLLLQOGOPMMRRRSRQRRRPMMNSSRSTTRRSOOOO AM:i:12 NM:i:2 SM:i:12 XM:i:2 XO:i:0 MQ:i:12 XT:A:M
	chomp;
	next if /^@/;
	my @t=split;
	next if $gender eq "F" and $t[2] eq "chrY";
	next if ($t[1]&256)==256;   #secondary alignment
	next if ($t[1]&2048)==2048;   #supplementary alignment
	my $current_len=length($t[9]);
	$max_len<$current_len and $max_len=$current_len;
	    $total_reads++;
#	if(($t[1]&4)==4){	#unmapped
#	    #$unmap_reads++;
#	}else{	#mapped

	if(($t[1]&4)!=4){	#mapped
	    $count_GC+=$t[9]=~tr/GCgc//;
	    $count_AT+=$t[9]=~tr/ATat//;
	    $count_N+=$t[9]=~tr/Nn//;
	    $map_reads++;
	    $duplication++ if ($t[1]&1024)==1024;
	    if(($t[1]&8)==0){	#next segment mapped
		$pair_map_reads++;
	    }
	    $indel_reads++ if $t[5]=~/[ID]/;
	    my $md="";
	    $md=$1 if $_=~/\bMD:Z:(\S+)\b/;
	    $md=~s/\^[ATGC]+//ig;
	    $mismatch_reads++ if $md=~/[ATGCatgc]/;
	    #say STDERR "DEBUG\t",$t[8];#"\t",$insert_size[$ins];
	    if($t[8]>0){
		my $ins=$t[8];
		$ins=2000 if $ins>2000;
		$insert_size[$ins]++;
		#say STDERR "\tDEBUG\t",$t[8],"\t",$insert_size[$ins];
	    }
	    my $maplen=0;
	    my $cigar=$t[5];
	    while($cigar=~s/(\d+)D//){
		    $maplen+=$1;
	    }
	    while($cigar=~s/(\d+)M//){
		    $maplen+=$1;
	    }
	    $read_depth+=$maplen;
	    $read_depth_rmdup+=$maplen unless ($t[1]&1024)==1024;;


	    if($bed){ # overlape reads
		my ($target_mark,$flank_mark)=(0,0);
		while($region[$r] and ($chr_ord{$t[2]}>$chr_ord{$region[$r][1]} or ($t[2] eq $region[$r][1] and $t[3]>$region[$r][3]))){
		    $r++;
		}#skip region without overlape
		foreach my $i (0..2){
		    if($region[$r+$i] and $t[2] eq $region[$r+$i][1] and ($t[3]<=$region[$r+$i][3] and $t[3]+$maplen-1>$region[$r+$i][2])){
			eval("\$$region[$r+$i][0]_mark++"); # mark overlape region
		    }
		}
		if($target_mark){
		    $target_map_reads++;
		    #########################################################################################
		    $target_map_dup++ if ($t[1]&1024)==1024;
		}elsif($flank_mark){
		    $flank_map_reads++;
		}
	    }

	    while($chr_ord{$t[2]} > $chr_ord{$chr}){	#meet a new chr
		if($chr ne "chr"){
		    if($bed){	#target region
			if(@left_region) {
			    region_depth(@left_region);  # unmaped chr  _depth[0]++ _read_depth+=0
			    @left_region=();
			    $d++;
			}else{
			    while($region[$d] and $region[$d][1] eq $chr){
				region_depth(@{$region[$d]}); # unmaped chr  _depth[0]++ _read_depth+=0
				$d++;
			    }
			}
		    }else{	#whole genome
			for(my $i=0; $i<$chr_len{$chr}-$offset; $i++){
			    if($i<$win){	#in the window
				$target_depth[$window[$i]]++;  # 1st target_depth[0]+=$win
				$target_depth_rmdup[$window_rmdup[$i]]++;  # 1st target_depth[0]+=$win
			    }else{	#exceed the window
				$target_depth[0]++; # 1st target_depth[0]+=chr_len-$win
				$target_depth_rmdup[0]++; # 1st target_depth[0]+=chr_len-$win
			    }
			} # umaped chr :  target_dept[0]+=$chr_len
		    }
		}
		#reset the offset
		$offset=0;
		for(my $i=0; $i<$win; $i++){
			$window[$i]=0;
			$window_rmdup[$i]=0;
		}
		$chr=$chr_chr{$chr_ord{$chr}+1};
	    }

	    while($t[2] eq $chr and $t[3]+$maplen-$offset-1>$win){	#exceed the window
		if($bed){	#target region
		    if(@left_region){
			if($left_region[3]<=$offset+$win-$len){
			    region_depth(@left_region); # unmap _depth[0]++ _read_depth+=0  maped _depth[n}++ _read_depth+=base
			    @left_region=();
			    $d++;
			    while($region[$d] and $region[$d][1] eq $chr and $region[$d][3]<=$offset+$win-$len){
				region_depth(@{$region[$d]}); #  unmap _depth[0]++ _read_depth+=0
				$d++;
			    }
			    if($region[$d] and $region[$d][1] eq $chr and $region[$d][2]<$offset+$win-$len){
				region_depth($region[$d][0],$region[$d][1],$region[$d][2],$offset+$win-$len); # unmap _depth[0]++ _red_depth+=0
				@left_region=($region[$d][0],$region[$d][1],$offset+$win-$len,$region[$d][3]);
			    }
			}else{
			    region_depth($left_region[0],$left_region[1],$left_region[2],$offset+$win-$len);
			    @left_region=($left_region[0],$left_region[1],$offset+$win-$len,$left_region[3]);
			}
		    }else{
			while($region[$d] and $region[$d][1] eq $chr and $region[$d][3]<=$offset+$win-$len){ 
			    region_depth(@{$region[$d]}); #before pos _depth[0]++ _read_depth+=0
			    $d++;
			}
			if($region[$d] and $region[$d][1] eq $chr and $region[$d][2]<$offset+$win-$len){
			    region_depth($region[$d][0],$region[$d][1],$region[$d][2],$offset+$win-$len); #1sh _depth[0]++ _read_depth+=0
			    @left_region=($region[$d][0],$region[$d][1],$offset+$win-$len,$region[$d][3]);
			}
		    }
		}else{	#whole genome
		    for(my $i=0; $i<$win-$len; $i++){
			$target_depth[$window[$i]]++; #unmap win  target_depth[0]+=$win-$len
			$target_depth_rmdup[$window_rmdup[$i]]++; #unmap win  target_depth[0]+=$win-$len
		    }
		}
		$offset+=($win-$len);
		for(my $i=0; $i<$len; $i++){
		    $window[$i]=$window[$i+$win-$len];
		    $window_rmdup[$i]=$window_rmdup[$i+$win-$len];
		}
		for(my $i=$len; $i<$win; $i++){
		    $window[$i]=0;
		    $window_rmdup[$i]=0;
		}
	    }

	    if(@left_region and $t[3]>$left_region[3]){
		    region_depth(@left_region); # unmap _depth[0]++ _read_depth+=0 maped _depth[n]++ _read_depth+=base
		    @left_region=();
		    $d++;
	    }
	    for(my $i=$t[3]; $i<$t[3]+$maplen; $i++){
		    $window[$i-$offset-1]++;   # map to windows
		    $window_rmdup[$i-$offset-1]++ unless ($t[1]&1024)==1024;
	    }
    }
}
if($bed){	#target region
    if(@left_region){
	    region_depth(@left_region); # unmap 
	    @left_region=();
	    $d++;
    }else{
	    while($region[$d] and $region[$d][1] eq $chr){
		    region_depth(@{$region[$d]}); # unmaped
		    $d++;
	    }
    }
}else{	#whole genome
    for(my $i=0; $i<$chr_len{$chr}-$offset; $i++){
	    if($i<$win){	#in the window
		    $target_depth[$window[$i]]++;
		    $target_depth_rmdup[$window_rmdup[$i]]++;
	    }else{	#exceed the window
		    $target_depth[0]++;
		    $target_depth_rmdup[0]++;
	    }
    }
}

while(exists $chr_chr{$chr_ord{$chr}+1}){	#the left unmapped chr
    $chr=$chr_chr{$chr_ord{$chr}+1};
    if($bed){	#target region
	    while($region[$d] and $region[$d][1] eq $chr){
		    eval("\$$region[$d][0]_depth[0]+=($region[$d][3]-$region[$d][2])");
		    eval("\$$region[$d][0]_depth_rmdup[0]+=($region[$d][3]-$region[$d][2])");
		    $d++;
	    }
    }else{	#whole genome
	    $target_depth[0]+=$chr_len{$chr};
	    $target_depth_rmdup[0]+=$chr_len{$chr};
    }
}

@target_cumu_depth=cumu_depth(@target_depth);
@target_cumu_depth_rmdup=cumu_depth(@target_depth_rmdup);

#stat file
open(STAT,">$output"."stat") || die "$!";
print STAT "Basic Information\n";
print STAT "ALL bases:\t",$total_reads*$max_len,"\n";###############################
print STAT "Mapped reads:\t$map_reads\n";
print STAT "Mapped reads[RM DUP]:\t",($map_reads-$duplication),"\n";
#############################################################################################################################################     
print STAT "Total mapped bases:\t$read_depth\n";
print STAT "Total mapped bases[GC,AT,N]:\t$count_GC,$count_AT,$count_N\n";
print STAT "Total mapped bases[RM DUP]:\t$read_depth_rmdup\n";####################################################
print STAT "Total mapped GC Rate:\t",FormatSigFigs($count_GC/($read_depth)*100,4)||0,"%\n";
print STAT "Mapping Rate:\t",FormatSigFigs($map_reads/$total_reads*100,4)||0,"%\n";
print STAT "Duplication Rate:\t",FormatSigFigs($duplication/$map_reads*100,4)||0,"%\n";
print STAT "Mismatch Reads Rate:\t",FormatSigFigs($mismatch_reads/$map_reads*100,4)||0,"%\n";
print STAT "Indel Reads Rate:\t",FormatSigFigs($indel_reads/$map_reads*100,4)||0,"%\n";
if($bed){
#	print STAT "\n";
    $target_cumu_depth[4]||=0;
    $target_cumu_depth[10]||=0;
    $target_cumu_depth[20]||=0;
    print STAT "Target Information\n";
    print STAT "Use Rate(target_read_depth_rmdup/All_base):\t",FormatSigFigs($target_read_depth_rmdup/($total_reads*$max_len)*100,4)||0,"%\n";##########################################
    print STAT "Target Size:\t$targetsize\n";
    print STAT "Mapped reads to target:\t$target_map_reads\n";###############################################################
    print STAT "Mapped reads to target[RM DUP]:\t",$target_map_reads-$target_map_dup,"\n";###############################################################
    print STAT "Capture specificity:\t",sprintf("%.3f",$target_map_reads/$map_reads*100),"%\n";
    print STAT "Capture specificity[RM DUP]:\t",sprintf("%.3f",($target_map_reads-$target_map_dup)/($map_reads-$duplication)*100),"%\n";
#############################################################################################################################################
    print STAT "Mapped bases to target:\t$target_read_depth\n";##########################################################################     
    print STAT "Mapped bases to target[RM DUP]:\t",$target_read_depth_rmdup,"\n";##########################################################################     
    print STAT "Capture Effiency:\t",sprintf("%.3f",$target_read_depth/$read_depth*100),"%\n";
    print STAT "Capture Effiency[RM DUP]:\t",sprintf("%.3f",$target_read_depth_rmdup/$read_depth_rmdup*100),"%\n";
#############################################################################################################################################    
    print STAT "Mapped Reads Number in Target:\t$target_map_reads\n";
#	print STAT "Data mapped to target region (Mb):\t",sprintf("%.2f",$target_read_depth/1000000),"\n";
#    print STAT "Target coverage:\t",FormatSigFigs($target_cumu_depth[1]/$targetsize*100,4)||0,"%\n";
    print STAT "Coverage of target region:\t",FormatSigFigs($target_cumu_depth[1]/$targetsize*100,4)||0,"%\n";
    print STAT "Target coverage >=4X percentage:\t",FormatSigFigs($target_cumu_depth[4]/$targetsize*100,4)||0,"%\n";
    print STAT "Target coverage >=10X percentage:\t",FormatSigFigs($target_cumu_depth[10]/$targetsize*100,4)||0,"%\n";
#    print STAT "Target coverage >=20X percentage:\t",FormatSigFigs($target_cumu_depth[20]/$targetsize*100,4)||0,"%\n";
    print STAT "Fraction of target covered >=20X:\t",FormatSigFigs($target_cumu_depth[20]/$targetsize*100,4)||0,"%\n";
#    print STAT "Target Mean Depth:\t",sprintf("%.3f",$target_read_depth/$targetsize)||0,"\n";
    print STAT "Mean depth of target region:\t",sprintf("%.3f",$target_read_depth/$targetsize)||0,"\n";
    $target_cumu_depth_rmdup[4]||=0;
    $target_cumu_depth_rmdup[10]||=0;
    $target_cumu_depth_rmdup[20]||=0;
    print STAT "Target coverage[RM DUP]:\t",FormatSigFigs($target_cumu_depth_rmdup[1]/$targetsize*100,4)||0,"%\n";
    print STAT "Target coverage >=4X percentage[RM DUP]:\t",FormatSigFigs($target_cumu_depth_rmdup[4]/$targetsize*100,4)||0,"%\n";
    print STAT "Target coverage >=10X percentage[RM DUP]:\t",FormatSigFigs($target_cumu_depth_rmdup[10]/$targetsize*100,4)||0,"%\n";
    print STAT "Target coverage >=20X percentage[RM DUP]:\t",FormatSigFigs($target_cumu_depth_rmdup[20]/$targetsize*100,4)||0,"%\n";
    print STAT "Target Mean Depth[RM DUP]:\t",sprintf("%.3f",$target_read_depth_rmdup/$targetsize)||0,"\n";
#	print STAT "Number of reads uniquely mapped to target:\t$target_uniq_map_reads\n";
    if($flank){
	    @flank_cumu_depth=cumu_depth(@flank_depth);
	    $flank_cumu_depth[4]||=0;
	    $flank_cumu_depth[10]||=0;
	    $flank_cumu_depth[20]||=0;
	    print STAT "Flanking size:\t",$target_cumu_depth[0]+$flank_cumu_depth[0],"\n";
	    print STAT "Reads mapped to flanking region:\t",$target_map_reads+$flank_map_reads,"\n";
	    #print STAT "Flanking coverage:\t",FormatSigFigs(($target_cumu_depth[1]+$flank_cumu_depth[1])/($target_cumu_depth[0]+$flank_cumu_depth[0])*100,4)||0,"%\n";
	    #print STAT "Flanking coverage >=4X percentage:\t",FormatSigFigs(($target_cumu_depth[4]+$flank_cumu_depth[4])/($target_cumu_depth[0]+$flank_cumu_depth[0])*100,4)||0,"%\n";
	    #print STAT "Flanking coverage >=10X percentage:\t",FormatSigFigs(($target_cumu_depth[10]+$flank_cumu_depth[10])/($target_cumu_depth[0]+$flank_cumu_depth[0])*100,4)||0,"%\n";
	    #print STAT "Flanking coverage >=20X percentage:\t",FormatSigFigs(($target_cumu_depth[20]+$flank_cumu_depth[20])/($target_cumu_depth[0]+$flank_cumu_depth[0])*100,4)||0,"%\n";
	    #print STAT "Falnking Mean Depth:\t",sprintf("%.3f",($target_read_depth+$flank_read_depth)/($target_cumu_depth[0]+$flank_cumu_depth[0])),"\n";
	    @flank_cumu_depth_rmdup=cumu_depth(@flank_depth_rmdup);
	    $flank_cumu_depth_rmdup[4]||=0;
	    $flank_cumu_depth_rmdup[10]||=0;
	    $flank_cumu_depth_rmdup[20]||=0;
	    print STAT "Flanking coverage[RM DUP]:\t",FormatSigFigs(($target_cumu_depth_rmdup[1]+$flank_cumu_depth_rmdup[1])/($target_cumu_depth_rmdup[0]+$flank_cumu_depth_rmdup[0])*100,4)||0,"%\n";
	    print STAT "Flanking coverage >=4X percentage[RM DUP]:\t",FormatSigFigs(($target_cumu_depth_rmdup[4]+$flank_cumu_depth_rmdup[4])/($target_cumu_depth_rmdup[0]+$flank_cumu_depth_rmdup[0])*100,4)||0,"%\n";
	    print STAT "Flanking coverage >=10X percentage[RM DUP]:\t",FormatSigFigs(($target_cumu_depth_rmdup[10]+$flank_cumu_depth_rmdup[10])/($target_cumu_depth_rmdup[0]+$flank_cumu_depth_rmdup[0])*100,4)||0,"%\n";
	    print STAT "Flanking coverage >=20X percentage[RM DUP]:\t",FormatSigFigs(($target_cumu_depth_rmdup[20]+$flank_cumu_depth_rmdup[20])/($target_cumu_depth_rmdup[0]+$flank_cumu_depth_rmdup[0])*100,4)||0,"%\n";
	    print STAT "Flanking Mean Depth[RM DUP]:\t",sprintf("%.3f",($target_read_depth_rmdup+$flank_read_depth_rmdup)/($target_cumu_depth_rmdup[0]+$flank_cumu_depth_rmdup[0])),"\n";
    }
}
close STAT;

open(CHR,">$output"."chr.depth") || die "$!";
print CHR "#chr\ttarget_size\tcoverage\tdepth\n";
for (sort {$a<=>$b} keys %chr_chr){
    if(exists $chr_size{$chr_chr{$_}}){
	    $chr_cover{$chr_chr{$_}}||=0;
	    $chr_depth{$chr_chr{$_}}||=0;
	    print CHR "$chr_chr{$_}\t$chr_size{$chr_chr{$_}}\t",sprintf("%.4f",$chr_cover{$chr_chr{$_}}/$chr_size{$chr_chr{$_}}),"\t",sprintf("%.2f",$chr_depth{$chr_chr{$_}}/$chr_size{$chr_chr{$_}}),"\n";
    }
}
close CHR;


#target region perBase depth file
open(PD,">$output"."perBase.depth") || die "$!";
for(my $i=0; $i<@target_depth; $i++){
    $target_depth[$i]||=0;
    print PD "$i\t$target_depth[$i]\t",FormatSigFigs($target_depth[$i]/$target_cumu_depth[0],4)||0,"\n";
}
close PD;

#target region Cumu depth file
open(CD,">$output"."Cumu.depth") || die "$!";
for(my $i=0; $i<@target_cumu_depth; $i++){
    $target_cumu_depth[$i]||=0;
    print CD "$i\t$target_cumu_depth[$i]\t",FormatSigFigs($target_cumu_depth[$i]/$target_cumu_depth[0],4)||0,"\n";
}
close CD;

#insert size file
open(IS,">$output"."insertsize") || die "$!";
for(my $i=1; $i<@insert_size; $i++){
    $insert_size[$i]||=0;
    print IS "$i\t",$insert_size[$i],"\t",FormatSigFigs($insert_size[$i]/$pair_map_reads*2,4)||0,"\n";
}
close IS;

#region depth statistics
sub region_depth{
    my @regn=@_;
    for(my $i=$regn[2]; $i<$regn[3]; $i++){
	    if($i-$offset<$win){	#in the window
		    eval("\$$regn[0]_depth[$window[$i-$offset]]++");
		    eval("\$$regn[0]_read_depth+=$window[$i-$offset]");
		    eval("\$$regn[0]_depth_rmdup[$window_rmdup[$i-$offset]]++");
		    eval("\$$regn[0]_read_depth_rmdup+=$window_rmdup[$i-$offset]");
		    if($regn[0] eq "target"){
			    $chr_size{$regn[1]}++;
			    $chr_depth{$regn[1]}+=$window[$i-$offset];
			    $chr_cover{$regn[1]}++ if $window[$i-$offset];
		    }
	    }else{	#exceed the window
		    eval("\$$regn[0]_depth[0]++");
		    eval("\$$regn[0]_depth_rmdup[0]++");
		    if($regn[0] eq "target"){
			    $chr_size{$regn[1]}++;
		    }
	    }
    }
}



#Cumu depth statistics
sub cumu_depth{
    my @regn_depth=@_;
    my @cumu_depth;
    $cumu_depth[scalar(@regn_depth)-1]=$regn_depth[scalar(@regn_depth)-1];
    for(my $i=scalar(@regn_depth)-2; $i>=0; $i--){
	    $regn_depth[$i]||=0;
	    $cumu_depth[$i]=$cumu_depth[$i+1]+$regn_depth[$i];
    }
    return @cumu_depth;
}
