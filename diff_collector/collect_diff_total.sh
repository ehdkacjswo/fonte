#!/bin/bash

# Set the base directory to iterate through
BASE_DIR=/root/workspace/data/Defects4J/baseline/

# Check if the base directory is provided and exists
if [[ -z "$BASE_DIR" || ! -d "$BASE_DIR" ]]; then
    echo "Please provide a valid directory."
    exit 1
fi

# Iterate through directories in the base directory
for dir in "$BASE_DIR"/*/; do
    # Get the directory name (basename strips off the path)
    dir_name=$(basename "$dir")

    # Use pattern matching to extract the relevant parts from the directory name
    if [[ $dir_name =~ ^([A-Za-z]+)-([0-9]+)([a-z]?)$ ]]; then
        # Extracted parts:
        name="${BASH_REMATCH[1]}"  # "Closure"
        number="${BASH_REMATCH[2]}"  # "21"
        optional_letter="${BASH_REMATCH[3]}"  # "b" (optional)
    else
        echo "Directory name $dir_name does not match the pattern."
    fi
    ./collect_diff.sh $name $number
done