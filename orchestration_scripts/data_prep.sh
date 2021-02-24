#!/bin/bash
#python -m src.preprocessing.detect_masks_without_staining
#python -m src.preprocessing.create_denoising_folds

for fold_i in {0..5}; do
  echo "Starting $fold_i .."
  bash orchestration_scripts/fold_embeddings_generation.sh $fold_i
  echo "Done $fold_i"
done