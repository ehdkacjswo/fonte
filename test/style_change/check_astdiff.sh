#!/bin/bash
analyzer_jar=/root/workspace/docker/workspace/tools/java_analyzer/target/java-analyzer-1.0-SNAPSHOT-shaded.jar

pid=$1
vid=$2
sha=$3
before_src_path=$4
after_src_path=$5
use_Rewrite=$6

workdir=/tmp/${pid}-${vid}b
[ ! -d $workdir ] && defects4j checkout -p ${pid} -v ${vid}b -w $workdir

cd $workdir
source $HOME/.sdkman/bin/sdkman-init.sh && sdk use java 11.0.12-open
echo "- Commit:" $sha
echo "-- File:" $after_src_path
# checkout to $sha
git checkout $sha $after_src_path
python /root/workspace/test/style_change/encoding_corrector.py -p $after_src_path
if [ "$use_Rewrite" = true ]; then
  java -cp $analyzer_jar analyzer.RewriteRunner $after_src_path cleanup.diff
  cp cleanup.diff "/root/workspace/cleanup"
  patch -p1 < cleanup.diff
  rm cleanup.diff
fi
cp $after_src_path $after_src_path.$sha
cp $after_src_path "/root/workspace/after_c.java"
# checkout to $sha~1
git checkout $sha~1 $before_src_path
python /root/workspace/test/style_change/encoding_corrector.py -p $before_src_path
cp $before_src_path "/root/workspace/before_ca.java"
if [ $? -eq 0 ]; then
  # only when $file is a valid path in $sha~1
  if [ "$use_Rewrite" = true ]; then
    java -cp $analyzer_jar analyzer.RewriteRunner $before_src_path cleanup.diff
    patch -p1 < cleanup.diff
    rm cleanup.diff
  fi
  cp $before_src_path $before_src_path.$sha~1
  cp $before_src_path "/root/workspace/before_c.java"

  # compare AST
  echo "start iso check"
  java -cp $analyzer_jar analyzer.ASTIsomorphicChecker \
    $after_src_path.$sha $before_src_path.$sha~1 > $after_src_path.$sha.is_isomorphic
  if [ $? -eq 0 ]; then
    if grep -Fxq "true" $after_src_path.$sha.is_isomorphic; then
      result='U'
    else
      result='C'
    fi
  else
    result='E' # error
  fi
  echo "--- ASTs are Isomorphic (w/ Rewrite: $use_Rewrite):" $(echo $after_src_path.$sha.is_isomorphic)

  rm $after_src_path.$sha
  rm $before_src_path.$sha~1

else
  rm $after_src_path.$sha

  echo "$before_src_path may not exist in $sha~1"
  echo "$before_src_path,$after_src_path,N" >> $output
fi

#echo "* Summary:" $output
#cat $output
echo ""
echo ""
