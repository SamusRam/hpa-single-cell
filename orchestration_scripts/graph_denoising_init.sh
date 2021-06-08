#!/bin/bash
for fold_i in {0..2}; do
  python -m src.denoising.graph_denoising --fold $fold_i
done

# manual mitotic selection
#python -m src.denoising.mitotic_spindle_neighbours.py --fold 0 --precomputed-knn-graph-path output/denoising_0_20210422_084444/knn_graph_200.pkl