import os
import cv2
import pandas as pd
import multiprocessing
from numpy.random import seed
seed(10)
from tqdm.auto import tqdm
import numpy as np
import pickle

from ..data.utils import get_train_df_ohe, get_public_df_ohe, get_masks_precomputed
from ..data.datasets import DataGeneneratorRGB
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--public-data", action='store_true')
parser.add_argument("--detection-threshold", type=int, default=150)

args = parser.parse_args()

PUBLIC_DATA_FLAG = args.public_data
DETECTION_THRESHOLD = args.detection_threshold
BATCH_SIZE = 8

PATH_TO_MASKS_ROOT = '../input/hpa_cell_mask_public/' if PUBLIC_DATA_FLAG else '../input/hpa_cell_mask/'
OUTPUT_PATH = '../input/negs_with_obvious_staining_public.pkl' if PUBLIC_DATA_FLAG else '../input/negs_with_obvious_staining.pkl'

num_cores = multiprocessing.cpu_count()

main_df = get_public_df_ohe() if PUBLIC_DATA_FLAG else get_train_df_ohe()


def get_cells_with_obvious_staining(img_paths, img_height=2048, img_width=2048,
                  batch_size=BATCH_SIZE, detection_threshold=DETECTION_THRESHOLD):

    data_gen = DataGeneneratorRGB(img_paths,
                               shuffle=False, batch_size=batch_size,
                               resized_height=img_height, resized_width=img_width)

    results = []
    for batch_i in range(len(img_paths)//batch_size + (1 if len(img_paths)%batch_size != 0 else 0)):

        images_batch = data_gen.__getitem__(batch_i)
        images_batch = images_batch[:len(img_paths) - batch_i*batch_size, :, :]
        img_batch_paths = img_paths[batch_i*batch_size:(batch_i + 1)*batch_size]
        try:
            masks_batch = get_masks_precomputed(img_batch_paths, PATH_TO_MASKS_ROOT)
        except ValueError:
            continue

        for mask_i, mask_init in enumerate(masks_batch):
            mask = cv2.resize(mask_init, (img_height, img_width))
            n_cells = mask.max()

            green_img = images_batch[mask_i][:, :, 1]
            for cell_i in range(1, n_cells + 1):
                cell_mask_bool = mask == cell_i
                cell_green_val = np.max(green_img[cell_mask_bool])

                if cell_green_val >= detection_threshold:
                    results.append(f'{os.path.basename(img_batch_paths[mask_i])}__{cell_i}')
    return results


negs_with_staining = []
inference_step = BATCH_SIZE

main_df = main_df.loc[main_df['Negative'] == 1]
for next_start_block_i in tqdm(range(0, main_df.shape[0], inference_step),
                               desc=('Detecting cells with apparent staining (rule-based) in '
                                       f'{"public" if PUBLIC_DATA_FLAG else "train"} data')):
    negs_with_staining.extend(get_cells_with_obvious_staining(main_df['img_base_path'].values[next_start_block_i: next_start_block_i + inference_step]))

with open(OUTPUT_PATH, 'wb') as f:
    pickle.dump(negs_with_staining, f)