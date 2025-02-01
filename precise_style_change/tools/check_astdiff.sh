#!/bin/bash
analyzer_jar=/root/workspace/docker/workspace/tools/java_analyzer/target/java-analyzer-1.0-SNAPSHOT-shaded.jar

pid=$1
vid=$2
commit_log=$3
sha=$4
use_Rewrite=$5
output=$6

[ -f $output ] && rm $output
output=$(realpath $output)

workdir=/tmp/${pid}-${vid}b
[ ! -d $workdir ] && defects4j checkout -p ${pid} -v ${vid}b -w $workdir

cd $workdir

python /root/workspace/precise_style_change/tools/get_touched_files.py $commit_log \
  --output $workdir/modified_files_${sha} \
  --commit $sha

echo "- Commit:" $sha
cat $workdir/modified_files_${sha} | while read after_src_path before_src_path; do
  #after_src_path=src/com/google/javascript/jscomp/NodeUtil.java
  #before_src_path=src/com/google/javascript/jscomp/NodeUtil.java
  echo "-- File:" $after_src_path
  # checkout to $sha
  git checkout $sha $after_src_path
  if [ "$use_Rewrite" = true ]; then
    astyle --mode=java --style=java $after_src_path
  fi
  cp $after_src_path /root/workspace/tmp/after.java
  # checkout to $sha~1
  git checkout $sha~1 $before_src_path
  if [ $? -eq 0 ]; then
    # only when $file is a valid path in $sha~1
    if [ "$use_Rewrite" = true ]; then
      astyle --mode=java --style=java $before_src_path
    fi
    cp $before_src_path /root/workspace/tmp/before.java

    # compare AST
    is_isomorphic=$(docker run --rm -v /home/coinse/doam/fonte/tmp:/diff gumtree isotest \
      -g java-jdtnc before.java after.java 2>&1)
    if [ $? -eq 0 ]; then
      case "$is_isomorphic" in
        "true") result='U' ;;
        "false") result='C' ;;
        *) result='E' ;;
      esac
    else
      result='E' # error
    fi
    echo "--- ASTs are Isomorphic (w/ Rewrite: $use_Rewrite):" $(echo $is_isomorphic)

    #rm $after_src_path.$sha
    #rm $before_src_path.$sha~1

    echo $before_src_path,$after_src_path,$result >> $output
  else
    #rm $after_src_path.$sha

    echo "$before_src_path may not exist in $sha~1"
    echo "$before_src_path,$after_src_path,N" >> $output
  fi
  #exit 1
done

#echo "* Summary:" $output
#cat $output
echo ""
echo ""
