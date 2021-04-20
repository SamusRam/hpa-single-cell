#!/bin/bash

echo "Trn data started.."
python -m src.preprocessing.generate_cell_bboxes > trn_bboxes.log

echo "Public data started.."
# for some reason it crashed without any error message (exhausted RAM and got killed by OS?), just cleaning the unfinished files with find . -type f -size 0 -delete and then restarting helped
python -m src.preprocessing.generate_cell_bboxes --public-data > public_bboxes_2.log

# to speed up on the second gpu
#python -m src.preprocessing.generate_cell_bboxes --public-data --gpu-id 1 --process-backward > public_bboxes_2_backward.log
