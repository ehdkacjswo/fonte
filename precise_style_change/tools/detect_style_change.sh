#!/bin/bash

pid=$1
vid=$2
tool=$3
use_Rewrite=$4

if [ "$use_Rewrite" = true ]; then
  scdir=/root/workspace/data/Defects4J/core/${pid}-${vid}b/$tool/validation
else
  scdir=/root/workspace/data/Defects4J/core/${pid}-${vid}b/$tool/validation_noOpenRewrite
fi

commits=/root/workspace/data/Defects4J/core/${pid}-${vid}b/$tool/commits.pkl

# clean up the output directory
[ ! -d $scdir ] && mkdir $scdir

# use Java 11
source $SDKMAN_DIR/bin/sdkman-init.sh && sdk use java 11.0.12-open

python /root/workspace/docker/workspace/tools/get_candidates.py $commits | while read sha; do
  if [ -f $scdir/$sha.csv ]; then
    #sha=147df44
    sh /root/workspace/precise_style_change/tools/check_astdiff.sh $pid $vid $commits $sha $use_Rewrite $scdir/$sha.csv
    #echo "$sha.csv exists"
  else
    sh /root/workspace/precise_style_change/tools/check_astdiff.sh $pid $vid $commits $sha $use_Rewrite $scdir/$sha.csv
  fi
done
