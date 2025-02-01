#!/bin/bash
error_trace=/root/workspace/setup.error

# Set the base directory to iterate through
BASE_DIR=/root/workspace/data/Defects4J/baseline/

# Check if the base directory is provided and exists
if [[ -z "$BASE_DIR" || ! -d "$BASE_DIR" ]]; then
  echo "Please provide a valid directory."
  exit 1
fi

project=${1:-"Closure"}
version=${2:-30}
target_commit=${3:-"e33e925"}
before_src_path=${4:-"src/com/google/javascript/jscomp/DefinitionsRemover.java"}
after_src_path=${5:-"src/com/google/javascript/jscomp/DefinitionsRemover.java"}
use_Rewrite=${6:-true}

original_cwd=$(pwd)

tmpdir=/tmp/${project}-${version}b/
faulty_version=$(grep ^${version}, /defects4j/framework/projects/${project}/commit-db | cut -d',' -f2)

# Checkout the faulty version to tmpdir
[ -d $tmpdir ] && rm -rf $tmpdir
defects4j checkout -p $project -v ${version}b -w ${tmpdir}
if [ ! -d $tmpdir ]; then
  echo "${project}-${version}b: [Error 1] checkout failure"
  exit 1
fi

cd $tmpdir
defects4j export -p dir.src.classes -o dir.src.classes

# Reset the commit to actual faulty version (Contents of the classes as well)
if [[ "$project" = "Time" ]] && (( $version > 20 )); then
  # Dataset doesn't contain such version
  git reset $faulty_version
  git checkout -- JodaTime/$(cat dir.src.classes)
  rm -rf $(cat dir.src.classes)
  mv JodaTime/$(cat dir.src.classes) $(cat dir.src.classes)
else
  git reset $faulty_version
  git checkout -- $(cat dir.src.classes)
  if [ $? -ne 0 ]; then
    echo "${project}-${version}b: [Error 2] no matching src dir"
    exit 2
  fi
fi
echo "Reset to the actual buggy version $faulty_version  OK"

# Collect relative diff
sh /root/workspace/test/style_change/check_astdiff.sh $project $version $target_commit $before_src_path $after_src_path $use_Rewrite

# Clean up the tmp directory
rm -rf $tmpdir
echo "Cleaning up $tmpdir"
cd "$original_cwd"
