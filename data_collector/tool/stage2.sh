#!/bin/bash
error_trace=/root/workspace/setup.error

# Set the base directory to iterate through
BASE_DIR=/root/workspace/data/Defects4J/baseline/

# Check if the base directory is provided and exists
if [[ -z "$BASE_DIR" || ! -d "$BASE_DIR" ]]; then
  echo "Please provide a valid directory."
  exit 1
fi

original_cwd=$(pwd)

# Iterate through directories in the base directory
for dir in "$BASE_DIR"/*/; do
  # Get the directory name (basename strips off the path)
  dir_name=$(basename "$dir")

  # Use pattern matching to extract the relevant parts from the directory name
  if [[ $dir_name =~ ^([A-Za-z]+)-([0-9]+)([a-z]?)$ ]]; then
    # Extracted parts:
    project="${BASH_REMATCH[1]}"  # "Closure"
    version="${BASH_REMATCH[2]}"  # "21"
    optional_letter="${BASH_REMATCH[3]}"  # "b" (optional)
  else
    echo "Directory name $dir_name does not match the pattern."
  fi

  project="Cli"
  version="29"

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
  echo "Reset to the actual buggy version $faulty_version  OK"

  # Collect relative diff
  python /root/workspace/diff_util/stage2.py -p $project -v $version

  # Clean up the tmp directory
  rm -rf $tmpdir
  echo "Cleaning up $tmpdir"
  cd "$original_cwd"
  exit 1
done

  
