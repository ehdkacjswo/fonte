#!/bin/bash
error_trace=/root/workspace/setup.error

# Set the base directory to iterate through
BASE_DIR=/root/workspace/data/Defects4J/baseline/

# Check if the base directory is provided and exists
if [[ -z "$BASE_DIR" || ! -d "$BASE_DIR" ]]; then
  echo "Please provide a valid directory."
  exit 1
fi

tool=${1:-"git"}
use_Rewrite=${2:-true}

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

  #project="Closure"
  #version="10"
  
  sh /root/workspace/lib/checkout.sh $project $version

  # Collect relative diff
  sh /root/workspace/precise_style_change/tools/detect_style_change.sh $project $version $tool $use_Rewrite
  sh /root/workspace/precise_style_change/tools/combine_style_change_results.sh $project $version $tool $use_Rewrite

  # Clean up the tmp directory
  #cd "$original_cwd"
  #exit 1
done

  
