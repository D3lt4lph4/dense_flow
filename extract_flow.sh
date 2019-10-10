#!/bin/bash

secs_to_human() {
    if [[ -z ${1} || ${1} -lt 60 ]] ;then
        min=0 ; secs="${1}"
    else
        time_mins=$(echo "scale=2; ${1}/60" | bc)
        min=$(echo ${time_mins} | cut -d'.' -f1)
        secs="0.$(echo ${time_mins} | cut -d'.' -f2)"
        secs=$(echo ${secs}*60|bc|awk '{print int($1+0.5)}')
    fi
    echo "Time Elapsed : ${min} minutes and ${secs} seconds."
}

files=$(find $1 -type f)

i=0

start=$(date +%s)
for file in $files
do
    # Getting the required variables
    filename=$(basename $file)
    filename="${filename%.*}"
    folder_name=$(basename $(dirname $file))

    output_dir=$2/$folder_name/$filename
    
    # Creating the directory to output the results
    mkdir -p $output_dir/flow_x
    mkdir -p $output_dir/flow_y
    mkdir -p $output_dir/image
    
    # Generating the files
    ./denseFlow_gpu --vidFile=$file -x=$output_dir/flow_x/flow_x -y=$output_dir/flow_y/flow_y -i=$output_dir/image/image  -b=20 -t=1 -d=0 -s=1 &

    i=$((i+1))

    # We start % thread at the same time.
    if ! ((i % 6)); then
    wait
    secs_to_human "$(($(date +%s) - ${start}))"
    fi
    

done
secs_to_human "$(($(date +%s) - ${start}))"