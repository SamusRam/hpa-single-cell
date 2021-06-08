import multiprocessing
import os

import cv2
from numpy.random import seed

seed(10)
from tqdm.auto import tqdm
import numpy as np
import pickle

from ..data.utils import get_train_df_ohe, get_public_df_ohe, get_masks_precomputed, get_cells_from_img
from ..data.datasets import DataGeneneratorRGB
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--public-data", action='store_true')
parser.add_argument("--detection-threshold", type=int, default=150)

args = parser.parse_args()

PUBLIC_DATA_FLAG = args.public_data
DETECTION_THRESHOLD = args.detection_threshold
BATCH_SIZE = 8

OUTPUT_PATH = 'input/negs_with_obvious_staining_public.pkl' if PUBLIC_DATA_FLAG else 'input/negs_with_obvious_staining.pkl'

num_cores = multiprocessing.cpu_count()

main_df = get_public_df_ohe(clean_from_duplicates=True,
                            clean_mitotic=True,
                            clean_aggresome=True) if PUBLIC_DATA_FLAG else get_train_df_ohe(clean_from_duplicates=True,
                                                                                            clean_mitotic=True,
                                                                                            clean_aggresome=True)


def get_cells_with_obvious_staining(img_paths, img_height=2048, img_width=2048,
                                    detection_threshold=DETECTION_THRESHOLD, n_top_pixels=20):
    results = []
    for base_path in tqdm(img_paths, desc=f'Detecting obvious negs..'):
        cell_imgs = get_cells_from_img(base_path, return_raw=True, target_img_size=img_height)

        for mask_i, cell_img in enumerate(cell_imgs):

            green_img = cell_img[:, :, 1]
            cell_i = mask_i + 1
            cell_green_val = np.mean(np.partition(green_img.flatten(), -10)[-10:])

            if cell_green_val >= detection_threshold:
                results.append(f'{os.path.basename(base_path)}__{cell_i}')
    return results


negs_with_staining = []
inference_step = BATCH_SIZE

main_df = main_df.loc[main_df['Negative'] == 1]
for next_start_block_i in tqdm(range(0, main_df.shape[0], inference_step),
                               desc=('Detecting cells with apparent staining (rule-based) in '
                                     f'{"public" if PUBLIC_DATA_FLAG else "train"} data')):
    negs_with_staining.extend(get_cells_with_obvious_staining(
        main_df['img_base_path'].values[next_start_block_i: next_start_block_i + inference_step]))

with open(OUTPUT_PATH, 'wb') as f:
    pickle.dump(negs_with_staining, f)
