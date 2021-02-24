#!/bin/bash
for fold_part in {0..5}; do
  nohup python -m src.preprocessing.generate_embeddings_bestfitting_per_fold --fold $1 --fold-part $fold_part --public-data &
  echo "Public part $fold_part put into background.."
done
python -m orchestration_scripts.active_waiting_for_empty_gpu
echo "Public data: done"

for fold_part in {0..5}; do
  nohup python -m src.preprocessing.generate_embeddings_bestfitting_per_fold --fold $1 --fold-part $fold_part &
  echo "Trn part $fold_part put into background.."
done
python -m orchestration_scripts.active_waiting_for_empty_gpu
echo "Trn data: done"
