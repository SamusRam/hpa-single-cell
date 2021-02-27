#!/bin/bash
for fold_i in {0..4}; do
  nohup python -m src.denoising.graph_denoising --fold $fold_i &
done