#!/bin/bash
#python -m src.preprocessing.detect_negatives_with_obvious_staining
#python -m src.preprocessing.detect_negatives_with_obvious_staining --public-data
#python -m src.preprocessing.create_imagelevel_folds
#python -m src.preprocessing.create_imagelevel_folds --obvious-staining

#python -m src.preprocessing.hpa-duplicate-images-in-train.py

#python -m src.preprocessing.create_denoising_folds

for fold_i in {0..4}; do
  echo "Starting $fold_i .."
  bash orchestration_scripts/fold_embeddings_generation.sh $fold_i
  echo "Done $fold_i"
done