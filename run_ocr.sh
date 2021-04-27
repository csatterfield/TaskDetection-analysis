#!/bin/bash
set x
for p in 'P14' 'P15' 'P16' 'P17' 'P18' 'P19';
    do
        python ocr.py ../archives/$p/Screenshots
        mv fulltext.pkl ../archives/$p
    done
