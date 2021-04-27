#!/bin/bash

file="$1"
outfile="${file%.*}"
convert -gravity SouthWest -crop 100x85%x+0+0 -units PixelsPerInch "$file" -resample 300 -type Grayscale "${outfile}.tiff"
tesseract -c language_model_penalty_non_dict_word=0.3 -c language_model_penalty_non_freq_dict_word=0.2 "${outfile}.tiff" "$outfile"
rm "${outfile}.tiff"
