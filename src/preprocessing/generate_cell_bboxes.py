import os
import cv2
import multiprocessing
from numpy.random import seed
seed(10)
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
import hpacellseg.cellsegmentator as cellsegmentator
from hpacellseg.utils import label_cell, label_nuclei
import warnings
warnings.simplefilter("ignore")
import logging
import pickle
from tqdm.auto import tqdm

from ..data.datasets import ProteinMLDatasetModified
from ..data.utils import get_public_df_ohe, get_train_df_ohe

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--public-data", action='store_true')
parser.add_argument('--gpu-id', default='0', type=str)
parser.add_argument('--process-backward', action='store_true')
parser.add_argument('--repair-zero-sized-masks', action='store_true')

args = parser.parse_args()

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(
    f'Bbox generation'
)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

BATCH_SIZE = 2
NUM_CORES = multiprocessing.cpu_count()
PUBLIC_DATA_FLAG = args.public_data
PATH_TO_MASKS_ROOT = '../input/hpa_cell_mask_public/' if PUBLIC_DATA_FLAG else '../input/hpa_cell_mask/'
OUTPUT_PATH = '../input/cell_bboxes_public' if PUBLIC_DATA_FLAG else '../input/cell_bboxes_train'
IMGS_FOLDER = '../input/publichpa_1024' if PUBLIC_DATA_FLAG else '../input/hpa-single-cell-image-classification/train'

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

main_df = get_public_df_ohe() if PUBLIC_DATA_FLAG else get_train_df_ohe()


NUC_MODEL = '../input/hpacellsegmentatormodelweights/dpn_unet_nuclei_v1.pth'
CELL_MODEL = '../input/hpacellsegmentatormodelweights/dpn_unet_cell_3ch_v1.pth'

segmentator = cellsegmentator.CellSegmentator(
    NUC_MODEL,
    CELL_MODEL,
    device='cuda',
    multi_channel_model=True,
    return_without_scale_restore=True
)


def get_masks(imgs):
    images = [[img[:, :, 0] for img in imgs],
              [img[:, :, 3] for img in imgs],
              [img[:, :, 2] for img in imgs]]

    nuc_segmentations, median_nuc_sizes = segmentator.pred_nuclei(images[2])
    cell_segmentations, init_sizes = segmentator.pred_cells(images, median_nuc_sizes=median_nuc_sizes)
    cell_masks = []
    for i in range(len(cell_segmentations)):
        cell_mask = label_cell(nuc_segmentations[i],
                               cell_segmentations[i],
                               median_nuc_sizes[i],
                               return_nuclei_label=False)
        cell_masks.append(cell_mask)

    return cell_masks


def store_cells(img_ids, folder=IMGS_FOLDER,
                       batch_size=BATCH_SIZE, num_workers=NUM_CORES//3):

    if args.repair_zero_sized_masks:
        img_ids = []
        # for file_name in tqdm(os.listdir(OUTPUT_PATH), desc='checking masks to repair..'):
        #     bboxes_path = os.path.join(OUTPUT_PATH, file_name)
        #     bboxes_df = pd.read_pickle(bboxes_path)
        #     for _, row in bboxes_df.iterrows():
        #         height = row['y_max'] - row['y_min']
        #         width = row['x_max'] - row['x_min']
        #         if min(height, width) <= 1:
        #             img_ids.append(file_name.replace('.pkl', ''))

        with open('../output/bbox_pred_inconsistent_basepaths.pkl', 'rb') as f:
            img_ids = set([os.path.basename(x) for x in pickle.load(f)])

        img_ids = list(set(img_ids))
        with open(f'../output/repaired_ids{"_public" if PUBLIC_DATA_FLAG else ""}.pkl', 'wb') as f:
            pickle.dump(img_ids, f)
        print(len(img_ids))
    else:
        img_ids = [img_id for img_id in img_ids if not os.path.exists(os.path.join(OUTPUT_PATH, f'{img_id}.pkl'))]

    print('ToDO', img_ids)
    dataset = ProteinMLDatasetModified(img_ids, folder=folder, resize=False)
    loader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=False, collate_fn=lambda x: x
    )

    def get_mask_index(cell_bool_mask, masks_all):
        cell_rows, cell_cols = np.where(cell_bool_mask)
        try:
            y_min, y_max = min(cell_rows), max(cell_rows)
            x_min, x_max = min(cell_cols), max(cell_cols)
            # y_min, y_max = [int(i) for i in np.quantile(cell_rows, [0.01, 0.99])]
            # x_min, x_max = [int(i) for i in np.quantile(cell_cols, [0.01, 0.99])]
        except TypeError:
            y_min = np.min(cell_rows)
            y_max = np.max(cell_rows)
            x_min = np.min(cell_cols)
            x_max = np.max(cell_cols)

        masks_all_bbox = masks_all[y_min: y_max, x_min:x_max]
        cell_rows_del, cell_cols_del = np.where(np.logical_and(np.logical_not(cell_bool_mask[y_min: y_max, x_min:x_max]),
                                                               masks_all_bbox))

        return [x_min, y_min], [x_max, y_max], cell_rows_del, cell_cols_del

    batch_i = -1
    for images_batch in loader:
        batch_i += 1
        if batch_i % 100 == 0:
            logger.info(f'{batch_i/len(loader)*100:.2f}%')
        images_batch = images_batch[:len(img_ids) - batch_i*batch_size]
        img_batch_ids = img_ids[batch_i*batch_size:(batch_i + 1)*batch_size]
        masks_batch = get_masks(images_batch)
        
        for mask_i, mask in enumerate(masks_batch):
            img_id = img_batch_ids[mask_i]
            pickle_path = os.path.join(OUTPUT_PATH, f'{img_id}.pkl')
            n_cells = mask.max()
            
            img_current = images_batch[mask_i]

            rows_list = []
            cols_list = []
            cell_i_list = []
            x_min_list = []
            x_max_list = []
            y_min_list = []
            y_max_list = []

            mask = cv2.resize(
                mask,
                (img_current.shape[0], img_current.shape[1]),
                interpolation=cv2.INTER_NEAREST,
            )
            masks_all = mask != 0
            for cell_i in (range(1, n_cells + 1)):
                cell_mask_bool = mask == cell_i
                [x_min, y_min], [x_max, y_max], cell_rows_del, cell_cols_del = get_mask_index(cell_mask_bool, masks_all)
                rows_list.append(cell_rows_del.astype(np.int16))
                cols_list.append(cell_cols_del.astype(np.int16))
                cell_i_list.append(np.int16(cell_i))
                x_min_list.append(np.int16(x_min))
                y_min_list.append(np.int16(y_min))
                x_max_list.append(np.int16(x_max))
                y_max_list.append(np.int16(y_max))

            results_df = pd.DataFrame({'x_min': x_min_list, 'y_min': y_min_list,
                                       'x_max': x_max_list, 'y_max': y_max_list,
                                       'cell_rows_del': rows_list, 'cell_cols_del': cols_list,
                                       'cell_i': cell_i_list
                                       })
            results_df.set_index('cell_i', inplace=True)
            results_df.to_pickle(pickle_path)


img_ids = main_df['ID'].values
if args.process_backward:
    img_ids = img_ids[::-1]

store_cells(img_ids)
