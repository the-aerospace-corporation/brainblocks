#!/usr/bin/env bash


declare -a filelist=("out/visual_test")

for i in "${filelist[@]}"
do
    # convert png files to video format
    ffmpeg -y -framerate 10 -i "${i}_frames_1_%05d.png"  -c:v libx264 -preset veryslow -crf 0 "${i}.mkv"

    # convert video to gif
    ffmpeg -y -i "${i}.mkv" -filter_complex "[0:v] split [a][b];[a] palettegen [p];[b][p] paletteuse" "${i}.gif"
done

