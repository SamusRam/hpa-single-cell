#!/bin/bash
python -m src.preprocessing.detect_masks_without_staining

python -m src.preprocessing.create_denoising_folds