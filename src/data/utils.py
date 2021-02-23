import os

import numpy as np
import pandas as pd


def get_class_names():
    # from https://www.kaggle.com/c/hpa-single-cell-image-classification/data
    specified_class_names = """0. Nucleoplasm
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

    class_names = [class_name.split('. ')[1] for class_name in specified_class_names.split('\n')]
    return class_names


def get_train_df_ohe(root_folder_path='../input/hpa-single-cell-image-classification', class_names=None):
    train_df = pd.read_csv(os.path.join(root_folder_path, 'train.csv'))
    train_df['img_base_path'] = train_df['ID'].map(lambda x: os.path.join(root_folder_path, x))
    # One-hot encoding classes
    train_df['Label'] = train_df['Label'].map(lambda x: map(int, x.split('|'))).map(set)
    if class_names is None:
        class_names = get_class_names()
    for class_i, class_name in enumerate(class_names):
        train_df[class_name] = train_df['Label'].map(lambda x: 1 if class_i in x else 0)
    return train_df


def get_public_df_ohe(public_info_df_path='../input/kaggle_2021.csv', class_names=None,
                      imgs_root_path='../input/publichpa_1024'):
    if class_names is None:
        class_names = get_class_names()

    public_hpa_df = pd.read_csv(public_info_df_path)
    celllines = ['A-431', 'A549', 'EFO-21', 'HAP1', 'HEK 293', 'HUVEC TERT2', 'HaCaT', 'HeLa', 'PC-3', 'RH-30',
                 'RPTEC TERT1', 'SH-SY5Y', 'SK-MEL-30', 'SiHa', 'U-2 OS', 'U-251 MG', 'hTCEpi']
    public_hpa_df_17 = public_hpa_df[public_hpa_df.Cellline.isin(celllines)]

    public_hpa_df_17['img_base_path'] = public_hpa_df_17['Image'].map(lambda x: os.path.join(imgs_root_path,
                                                                                             os.path.basename(x)))
    public_hpa_df_17['Label'] = public_hpa_df_17['Label'].map(lambda x: x.split(',')).map(set)
    for class_i, class_name in enumerate(class_names):
        public_hpa_df_17[class_name] = public_hpa_df_17['Label'].map(lambda x: 1 if class_name.strip() in x else 0).map(
            int)
        if 'Vesicles' in class_name:
            public_hpa_df_17[class_name] = public_hpa_df_17['Label'].map(lambda x: 1 if 'Vesicles' in x else 0).map(int)
        if 'Negative' in class_name:
            public_hpa_df_17[class_name] = public_hpa_df_17['Label'].map(lambda x: 1 if 'No staining' in x else 0).map(
                int)

    public_hpa_df_17 = public_hpa_df_17[((public_hpa_df_17['Nucleoplasm'] == 0) &
                                         (public_hpa_df_17['Cytosol'] == 0)) |
                                        (public_hpa_df_17['Aggresome'] == 1) |
                                        (public_hpa_df_17['Mitotic spindle'] == 1)]

    def are_all_imgs_present(base_path):
        for color in ['red', 'green', 'blue', 'yellow']:
            if not os.path.exists(f'{base_path}_{color}.png'):
                return False
        return True

    public_hpa_df_17 = public_hpa_df_17[public_hpa_df_17['img_base_path'].map(lambda x: are_all_imgs_present(x))]

    return public_hpa_df_17


def get_masks_precomputed(img_paths, masks_root):
    cell_masks = [np.load(f'{masks_root}/{os.path.basename(image_path)}.npz')['arr_0'] for image_path in img_paths]
    return cell_masks
