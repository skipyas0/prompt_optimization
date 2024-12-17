#!/bin/bash
model=""
# parse arguments
for i in "$@"; do
  if [[ $i == "--conf" ]]; then
    config_args=("${@:2:3}")
    model=${config_args[2]}
    break
  elif [[ $i == "--ident" ]]; then
    ident=$2
    IFS='-' read -ra ident_parts <<< "$ident"
    model=${ident_parts[2]}  # model is the 3rd part (index 2)
    break
  fi
done


echo $model
