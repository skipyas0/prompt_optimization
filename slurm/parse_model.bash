#!/bin/bash
model=""
# parse arguments
for i in "$@"; do
  #echo $i
  if [[ $i == "--conf" ]]; then
    # collect the next three arguments as config
    #echo "hit conf"
    config_args=("${@:2:3}")
    model=${config_args[2]}
    break
  elif [[ $i == "--ident" ]]; then
    #echo "hit ident"
    ident=$2
    IFS='-' read -ra ident_parts <<< "$ident"
    model=${ident_parts[2]}  # model is the 3rd part (index 2)
    break
  fi
done


echo $model
