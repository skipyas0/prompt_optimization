#!/bin/bash
# Check if a file is provided as an argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <json-file>"
    exit 1
fi

# Read the JSON file provided as the first argument
FILE_NAME="$1"
JSON_FILE="/home/kloudvoj/devel/prompt_optimization/conf/_model/${FILE_NAME}.json"
# Check if the file exists
if [ ! -f "$JSON_FILE" ]; then
    echo "File not found: $JSON_FILE"
    exit 1
fi

# Extract the "model" field using jq
MODEL=$(/usr/bin/jq -r '.model' "$JSON_FILE")

# Check if the field is present
if [ "$MODEL" == "null" ]; then
    echo "'model' field not found in $JSON_FILE"
    exit 1
fi

echo "$MODEL"
