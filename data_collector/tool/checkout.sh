#!/bin/bash
original_cwd=$(pwd)
project=$1
version=$2

tmpdir=/tmp/${project}-${version}b/
faulty_version=$(grep ^${version}, /defects4j/framework/projects/${project}/commit-db | cut -d',' -f2)
#source $HOME/.sdkman/bin/sdkman-init.sh && sdk use java 8.0.302-open

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
#echo "Reset to the actual buggy version $faulty_version  OK"

# Clean up the tmp directory
#rm -rf $tmpdir
#echo "Cleaning up $tmpdir"

  
