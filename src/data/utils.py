import os

import numpy as np
import pandas as pd
import cv2


SPECIFIED_CLASS_NAMES = """0. Nucleoplasm
    1. Nuclear membrane
    2. Nucleoli
    3. Nucleoli fibrillar center
    4. Nuclear speckles
    5. Nuclear bodies
    6. Endoplasmic reticulum
    7. Golgi apparatus
    8. Intermediate filaments
    9. Actin filaments 
    10. Microtubules
    11. Mitotic spindle
    12. Centrosome
    13. Plasma membrane
    14. Mitochondria
    15. Aggresome
    16. Cytosol
    17. Vesicles and punctate cytosolic patterns
    18. Negative"""


def get_class_names():
    class_names = [class_name.split('. ')[1] for class_name in SPECIFIED_CLASS_NAMES.split('\n')]
    return class_names


def get_train_df_ohe(root_folder_path='../input/hpa-single-cell-image-classification', class_names=None):
    train_df = pd.read_csv(os.path.join(root_folder_path, 'train.csv'))
    train_df['img_base_path'] = train_df['ID'].map(lambda x: os.path.join(root_folder_path, 'train', x))
    # One-hot encoding classes
    train_df['Label'] = train_df['Label'].map(lambda x: map(int, x.split('|'))).map(set)
    if class_names is None:
        class_names = get_class_names()
    for class_i, class_name in enumerate(class_names):
        train_df[class_name] = train_df['Label'].map(lambda x: 1 if class_i in x else 0)
    return train_df[['ID', 'img_base_path'] + class_names]


def are_all_imgs_present(base_path):
    for color in ['red', 'green', 'blue', 'yellow']:
        if not os.path.exists(f'{base_path}_{color}.png'):
            return False
    return True


def get_public_df_ohe(public_info_df_path='../input/kaggle_2021.tsv', class_names=None,
                      imgs_root_path='../input/publichpa_1024'):
    if class_names is None:
        class_names = get_class_names()

    public_hpa_df = pd.read_csv(public_info_df_path)
    # Remove all images overlapping with Training set
    public_hpa_df = public_hpa_df[~public_hpa_df.in_trainset]

    # Remove all images with only labels that are not in this competition
    public_hpa_df = public_hpa_df[~public_hpa_df.Label_idx.isna()]

    celllines = ['A-431', 'A549', 'EFO-21', 'HAP1', 'HEK 293', 'HUVEC TERT2', 'HaCaT', 'HeLa', 'PC-3', 'RH-30',
                 'RPTEC TERT1', 'SH-SY5Y', 'SK-MEL-30', 'SiHa', 'U-2 OS', 'U-251 MG', 'hTCEpi']
    public_hpa_df_17 = public_hpa_df[public_hpa_df.Cellline.isin(celllines)].copy()
    public_hpa_df_17['img_base_path'] = public_hpa_df_17['Image'].map(lambda x: os.path.join(imgs_root_path,
                                                                                             os.path.basename(x)))

    public_hpa_df_17['Label_idx'] = public_hpa_df_17['Label_idx'].map(lambda x: map(int, x.split('|'))).map(set)

    for class_i, class_name in enumerate(class_names):
        public_hpa_df_17[class_name] = public_hpa_df_17['Label_idx'].map(lambda x: 1 if class_i in x else 0)

    public_hpa_df_17 = public_hpa_df_17[public_hpa_df_17['img_base_path'].map(lambda x: are_all_imgs_present(x))]

    public_hpa_df_17.columns = [x if x != 'Image' else 'ID' for x in public_hpa_df_17.columns]
    public_hpa_df_17['ID'] = public_hpa_df_17['ID'].map(lambda x: x.split('/')[-1])

    forbidden_ids = {'1835_D1_3'}
    public_hpa_df_17 = public_hpa_df_17[~public_hpa_df_17['ID'].isin(forbidden_ids)]
    return public_hpa_df_17[['ID', 'img_base_path'] + class_names]


def get_masks_precomputed(img_paths, masks_root):
    cell_masks = [np.load(f'{masks_root}/{os.path.basename(image_path)}.npz')['arr_0'] for image_path in img_paths]
    return cell_masks


def open_rgb(image_id,
              folder_root='../input/hpa-single-cell-image-classification/train'):  # a function that reads RGB image
    colors = ['red', 'green', 'blue']
    img = [cv2.imread(f'{folder_root}/{image_id}_{color}.png', cv2.IMREAD_GRAYSCALE)
           for color in colors]
    img = np.stack(img, axis=-1)
    return img


def get_new_class_name_indices_in_prev_comp_data():

    old_comp_specified_class_names = """0.  Nucleoplasm  
    1.  Nuclear membrane   
    2.  Nucleoli   
    3.  Nucleoli fibrillar center   
    4.  Nuclear speckles   
    5.  Nuclear bodies   
    6.  Endoplasmic reticulum   
    7.  Golgi apparatus   
    8.  Peroxisomes   
    9.  Endosomes   
    10.  Lysosomes   
    11.  Intermediate filaments   
    12.  Actin filaments   
    13.  Focal adhesion sites   
    14.  Microtubules   
    15.  Microtubule ends   
    16.  Cytokinetic bridge   
    17.  Mitotic spindle   
    18.  Microtubule organizing center   
    19.  Centrosome   
    20.  Lipid droplets   
    21.  Plasma membrane   
    22.  Cell junctions   
    23.  Mitochondria   
    24.  Aggresome   
    25.  Cytosol   
    26.  Cytoplasmic bodies   
    27.  Rods & rings  """

    class_name_2_new_idx = {"Nucleoplasm": 0,
    "Nuclear membrane": 1,
    "Nucleoli": 2,
    "Nucleoli fibrillar center": 3,
    "Nuclear speckles": 4,
    "Nuclear bodies": 5,
    "Endoplasmic reticulum": 6,
    "Golgi apparatus": 7,
    "Intermediate filaments": 8,
    "Actin filaments": 9,
    "Focal adhesion sites": 9,
    "Microtubules": 10,
    "Mitotic spindle": 11,
    "Centrosome": 12,
    "Centriolar satellite": 12,
    "Plasma membrane": 13,
    "Cell Junctions": 13,
    "Mitochondria": 14,
    "Aggresome": 15,
    "Cytosol": 16,
    "Vesicles": 17,
    "Peroxisomes": 17,
    "Endosomes": 17,
    "Lysosomes": 17,
    "Lipid droplets": 17,
    "Cytoplasmic bodies": 17,
    "No staining": 18}

    old_2_new_indices = {class_i_old: class_name_2_new_idx[name]
                         for class_i_old, name in enumerate(old_comp_specified_class_names)
                         if name in class_name_2_new_idx}

    new_name_index_2_old_name_index = dict()
    for old_name_index, new_class_index in old_2_new_indices.items():
        if new_class_index in new_name_index_2_old_name_index:
            new_name_index_2_old_name_index[new_class_index].append(old_name_index)
        else:
            new_name_index_2_old_name_index[new_class_index] = [old_name_index]
    return new_name_index_2_old_name_index