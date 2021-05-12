#!/bin/bash
#python -m src.preprocessing.detect_negatives_with_obvious_staining
#python -m src.preprocessing.detect_negatives_with_obvious_staining --public-data
#python -m src.preprocessing.create_imagelevel_folds
#python -m src.preprocessing.create_imagelevel_folds --obvious-staining

#python -m src.preprocessing.hpa-duplicate-images-in-train

# python -m src.predict.predict_cells_from_image_level_densenet # I've manually parallelized per available GPU's, e.g., nohup python -m src.predict.predict_cells_from_image_level_densenet --fold-single 1 --fold-one-fifth-number 0 > image_level_pred_1_0.log &
# due to concerns that I might run out of RAM, I created a separate h5 file for each image, it has significant disk usage overhead, can be optimized
# python -m src.preprocessing.unify_predictions_from_image_level_densenet
# python -m src.preprocessing.unify_embeddings_from_image_level_densenet

#python -m src.preprocessing.create_denoising_folds
