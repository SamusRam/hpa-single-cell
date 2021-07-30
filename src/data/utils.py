import os

import cv2
import numpy as np
import pandas as pd

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


def get_train_df_ohe(root_folder_path='input/hpa-single-cell-image-classification', class_names=None,
                     clean_from_duplicates=False, clean_mitotic=False, clean_aggresome=False):
    train_df = pd.read_csv(os.path.join(root_folder_path, 'train.csv'))

    train_df['img_base_path'] = train_df['ID'].map(lambda x: os.path.join(root_folder_path, 'train', x))
    # One-hot encoding classes
    train_df['Label'] = train_df['Label'].map(lambda x: map(int, x.split('|'))).map(set)
    if class_names is None:
        class_names = get_class_names()
    for class_i, class_name in enumerate(class_names):
        train_df[class_name] = train_df['Label'].map(lambda x: 1 if class_i in x else 0)

    if clean_from_duplicates:
        duplicates = pd.read_csv('output/duplicates.csv.gz')
        forbidden_basepaths = set(duplicates['Extra'].values)
        for needed_id in ['5d36256a-bbbe-11e8-b2ba-ac1f6b6435d0',
                          '68d5cd28-bbc6-11e8-b2bc-ac1f6b6435d0',
                          '96427802-bbac-11e8-b2ba-ac1f6b6435d0',
                          '1469d230-bbc5-11e8-b2bc-ac1f6b6435d0',
                          '78411ae2-bbc6-11e8-b2bc-ac1f6b6435d0',
                          '14b5422c-bbbd-11e8-b2ba-ac1f6b6435d0']:
            forbidden_basepaths.remove(f'input/hpa-single-cell-image-classification/train/{needed_id}')
        train_df = train_df[~train_df['img_base_path'].isin(forbidden_basepaths)]

    if clean_mitotic:
        cherrypicked_mitotic_spindle = pd.read_csv('input/mitotic_cells_selection.csv')
        checked_mitotic_ids = set(cherrypicked_mitotic_spindle['ID'].values)
        train_df = train_df[(train_df['Mitotic spindle'] == 0) | (train_df['ID'].isin(checked_mitotic_ids))]

    if clean_aggresome:
        aggresome_blacklist_ids = set(pd.read_csv('input/aggresome_blacklist.csv')['ID'].values)
        train_df = train_df[(train_df['Aggresome'] == 0) | np.logical_not(train_df['ID'].isin(aggresome_blacklist_ids))]
    return train_df[['ID', 'img_base_path'] + class_names]


def are_all_imgs_present(base_path):
    for color in ['red', 'green', 'blue', 'yellow']:
        if not os.path.exists(f'{base_path}_{color}.png'):
            return False
    return True


def get_public_df_ohe(public_info_df_path='input/kaggle_2021.tsv', class_names=None,
                      imgs_root_path='input/publichpa_1024', clean_from_duplicates=False,
                      clean_mitotic=False, clean_aggresome=False):
    if class_names is None:
        class_names = get_class_names()

    public_hpa_df = pd.read_csv(public_info_df_path)
    # Remove all images overlapping with Training set
    public_hpa_df = public_hpa_df[~public_hpa_df.in_trainset]

    # Remove all images with only labels that are not in this competition
    public_hpa_df = public_hpa_df[~public_hpa_df.Label_idx.isna()]

    celllines = ['A-431', 'A549', 'EFO-21', 'HAP1', 'HEK 293', 'HUVEC TERT2', 'HaCaT', 'HeLa', 'PC-3', 'RH-30',
                 'RPTEC TERT1', 'SH-SY5Y', 'SK-MEL-30', 'SiHa', 'U-2 OS', 'U-251 MG', 'hTCEpi']

    public_hpa_df['Label_idx'] = public_hpa_df['Label_idx'].map(lambda x: map(int, x.split('|'))).map(set)

    for class_i, class_name in enumerate(class_names):
        public_hpa_df[class_name] = public_hpa_df['Label_idx'].map(lambda x: 1 if class_i in x else 0)

    public_hpa_df_17 = public_hpa_df[public_hpa_df.Cellline.isin(celllines) |
                                     (public_hpa_df['Mitotic spindle'] == 1) |
                                     (public_hpa_df['Aggresome'] == 1)].copy()
    public_hpa_df_17['img_base_path'] = public_hpa_df_17['Image'].map(lambda x: os.path.join(imgs_root_path,
                                                                                             os.path.basename(x)))

    public_hpa_df_17 = public_hpa_df_17[public_hpa_df_17['img_base_path'].map(lambda x: are_all_imgs_present(x))]

    public_hpa_df_17.columns = [x if x != 'Image' else 'ID' for x in public_hpa_df_17.columns]
    public_hpa_df_17['ID'] = public_hpa_df_17['ID'].map(lambda x: x.split('/')[-1])

    forbidden_ids = {'1835_D1_3'}
    public_hpa_df_17 = public_hpa_df_17[~public_hpa_df_17['ID'].isin(forbidden_ids)]
    if clean_from_duplicates:
        duplicates = pd.read_csv('output/duplicates.csv.gz')
        forbidden_basepaths = set(duplicates['Extra'].values)
        public_hpa_df_17 = public_hpa_df_17[~public_hpa_df_17['img_base_path'].isin(forbidden_basepaths)]

    if clean_mitotic:
        cherrypicked_mitotic_spindle = pd.read_csv('input/mitotic_cells_selection.csv')
        checked_mitotic_ids = set(cherrypicked_mitotic_spindle['ID'].values)
        public_hpa_df_17 = public_hpa_df_17[
            (public_hpa_df_17['Mitotic spindle'] == 0) | (public_hpa_df_17['ID'].isin(checked_mitotic_ids))]

    if clean_aggresome:
        aggresome_blacklist_ids = set(pd.read_csv('input/aggresome_blacklist.csv')['ID'].values)
        public_hpa_df_17 = public_hpa_df_17[
            (public_hpa_df_17['Aggresome'] == 0) | np.logical_not(public_hpa_df_17['ID'].isin(aggresome_blacklist_ids))]

    return public_hpa_df_17[['ID', 'img_base_path'] + class_names]


def get_masks_precomputed(img_paths, masks_root):
    raise NotImplementedError('npz masks not supported after segmentation update')
    # cell_masks = [np.load(f'{masks_root}/{os.path.basename(image_path)}.npz')['arr_0'] for image_path in img_paths]
    # return cell_masks


def get_cells_from_img(img_base_path, base_trn_path='input/hpa-single-cell-image-classification/train',
                       base_public_path='input/publichpa_1024',
                       trn_cell_boxes_path='input/cell_bboxes_train',
                       public_cell_boxes_path='input/cell_bboxes_public',
                       cell_img_size=512, return_raw=False, sample_size=None, cell_labels_df=None,
                       target_img_size=None):
    assert not return_raw or target_img_size is not None, 'when returning_raw target_img_size must be specified'

    is_from_train = 'train' in img_base_path
    cell_boxes_path = trn_cell_boxes_path if is_from_train else public_cell_boxes_path
    img_id = os.path.basename(img_base_path)
    bboxes_path = os.path.join(cell_boxes_path, f'{img_id}.pkl')
    bboxes_df = pd.read_pickle(bboxes_path)

    img_rgby = open_rgby(img_id, folder_root=base_trn_path if is_from_train else base_public_path)

    if return_raw:
        scale_factor = target_img_size / img_rgby.shape[0]

    cell_imgs = []
    if cell_labels_df is not None:
        cell_labels = []

    if sample_size is not None and sample_size < len(bboxes_df):
        iterator = bboxes_df.sample(sample_size).iterrows()
    else:
        iterator = bboxes_df.iterrows()

    for cell_i, row in iterator:
        img_cell = img_rgby[row['y_min']:row['y_max'], row['x_min']:row['x_max'], :].copy()
        img_cell[row['cell_rows_del'], row['cell_cols_del'], :] = 0

        if return_raw:
            required_shape = (int(img_cell.shape[1] * scale_factor), int(img_cell.shape[0] * scale_factor))
            img_cell = cv2.resize(img_cell, required_shape)
            yield img_cell
            continue

        height = row['y_max'] - row['y_min']
        width = row['x_max'] - row['x_min']
        if min(height, width) < 0.5 * max(height, width):
            if height < width:
                img_cell = np.tile(img_cell, [2, 1, 1, ])
            else:
                img_cell = np.tile(img_cell, [1, 2, 1])

        if img_cell.shape[0] > img_cell.shape[1]:
            diff = img_cell.shape[0] - img_cell.shape[1]
            left = diff // 2
            right = diff - left
            img_cell = cv2.copyMakeBorder(img_cell, 0, 0, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])
        else:
            diff = img_cell.shape[1] - img_cell.shape[0]
            up = diff // 2
            down = diff - up
            img_cell = cv2.copyMakeBorder(img_cell, up, down, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])

        if cell_img_size is not None:
            img_cell = cv2.resize(img_cell, (cell_img_size, cell_img_size))

        if cell_labels_df is not None:
            if cell_i - 1 in cell_labels_df.index.get_level_values(0):
                yield img_cell, cell_labels_df.loc[cell_i - 1].values[0]
                # cell_labels.append(cell_labels_df.loc[cell_i - 1].values[0])
                # cell_imgs.append(img_cell)
        else:
            yield img_cell
            # cell_imgs.append(img_cell)

    # if cell_labels_df is None:
    #     return cell_imgs
    # return cell_imgs, cell_labels


# TODO: refactor get_cell_img, get_cells_from_img, get_cell_img_with_mask
def get_cell_img(img_base_path, cell_i, base_trn_path='input/hpa-single-cell-image-classification/train',
                 base_public_path='input/publichpa_1024',
                 trn_cell_boxes_path='input/cell_bboxes_train',
                 public_cell_boxes_path='input/cell_bboxes_public',
                 cell_img_size=512, aug=None, target_raw_img_size=None):
    " cell_i must be 0-based "

    img_id = os.path.basename(img_base_path)
    is_from_train = len(img_id) > 15
    cell_boxes_path = trn_cell_boxes_path if is_from_train else public_cell_boxes_path
    bboxes_path = os.path.join(cell_boxes_path, f'{img_id}.pkl')
    bboxes_df = pd.read_pickle(bboxes_path)

    img_rgby = open_rgby(img_id, folder_root=base_trn_path if is_from_train else base_public_path)

    row = bboxes_df.loc[cell_i + 1]
    img_cell = img_rgby[row['y_min']:row['y_max'], row['x_min']:row['x_max'], :].copy()
    img_cell[row['cell_rows_del'], row['cell_cols_del'], :] = 0

    if aug is not None:
        img_cell = aug(img_cell)
    height, width = img_cell.shape[:2]

    if min(height, width) < 0.5 * max(height, width):
        if height < width:
            img_cell = np.tile(img_cell, [2, 1, 1, ])
        else:
            img_cell = np.tile(img_cell, [1, 2, 1])

    if img_cell.shape[0] > img_cell.shape[1]:
        diff = img_cell.shape[0] - img_cell.shape[1]
        left = diff // 2
        right = diff - left
        img_cell = cv2.copyMakeBorder(img_cell, 0, 0, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])
    else:
        diff = img_cell.shape[1] - img_cell.shape[0]
        up = diff // 2
        down = diff - up
        img_cell = cv2.copyMakeBorder(img_cell, up, down, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])

    if target_raw_img_size is not None:
        prescale_factor = target_raw_img_size / img_rgby.shape[0]
        if prescale_factor != 1:
            current_shape = img_cell.shape[:2]
            target_raw_size = int(prescale_factor*current_shape[0])
            img_cell = cv2.resize(img_cell, (target_raw_size, target_raw_size))
            if cell_img_size is not None:
                img_cell = cv2.resize(img_cell, (cell_img_size, cell_img_size))
            else:
                img_cell = cv2.resize(img_cell, current_shape)
        elif cell_img_size is not None:
            img_cell = cv2.resize(img_cell, (cell_img_size, cell_img_size))

    elif cell_img_size is not None:
        img_cell = cv2.resize(img_cell, (cell_img_size, cell_img_size))
    return img_cell


def get_cell_img_mitotic(img_base_path, cell_i, base_trn_path='input/hpa-single-cell-image-classification/train',
                 base_public_path='input/publichpa_1024',
                 trn_cell_boxes_path='input/cell_bboxes_train',
                 public_cell_boxes_path='input/cell_bboxes_public',
                 cell_img_size=224, aug=None, target_raw_img_size=None):
    " cell_i must be 0-based "

    img_id = os.path.basename(img_base_path)
    is_from_train = len(img_id) > 15
    cell_boxes_path = trn_cell_boxes_path if is_from_train else public_cell_boxes_path
    bboxes_path = os.path.join(cell_boxes_path, f'{img_id}.pkl')
    bboxes_df = pd.read_pickle(bboxes_path)

    img_rgby = open_rgby(img_id, folder_root=base_trn_path if is_from_train else base_public_path)

    row = bboxes_df.loc[cell_i + 1]
    y_min = row['y_min']
    y_max = row['y_max']
    x_min = row['x_min']
    x_max = row['x_max']

    Y_min = max(0, y_min - (y_max - y_min)//2)
    Y_max = y_max + (y_max - y_min)//2
    X_min = max(0, x_min - (x_max - x_min) // 2)
    X_max = x_max + (x_max - x_min) // 2

    img_cell = img_rgby[y_min:y_max, x_min:x_max, :].copy()
    img_cell[row['cell_rows_del'], row['cell_cols_del'], :] = img_cell[row['cell_rows_del'], row['cell_cols_del'], :]/3
    img_center_row = np.concatenate((img_rgby[y_min:y_max, X_min:x_min, :]/3,
                                     img_cell,
                                     img_rgby[y_min:y_max, x_max:X_max, :]/3), axis=1)
    img_cell = np.concatenate((img_rgby[Y_min: y_min, X_min:X_max]/3,
                               img_center_row,
                               img_rgby[y_max: Y_max, X_min:X_max]/3), axis=0)

    if aug is not None:
        img_cell = aug(img_cell)

    if img_cell.shape[0] > img_cell.shape[1]:
        diff = img_cell.shape[0] - img_cell.shape[1]
        left = diff // 2
        right = diff - left
        img_cell = cv2.copyMakeBorder(img_cell, 0, 0, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])
    else:
        diff = img_cell.shape[1] - img_cell.shape[0]
        up = diff // 2
        down = diff - up
        img_cell = cv2.copyMakeBorder(img_cell, up, down, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])

    if target_raw_img_size is not None:
        prescale_factor = target_raw_img_size / img_rgby.shape[0]
        if prescale_factor != 1:
            current_shape = img_cell.shape[:2]
            target_raw_size = int(prescale_factor*current_shape[0])
            img_cell = cv2.resize(img_cell, (target_raw_size, target_raw_size))
            if cell_img_size is not None:
                img_cell = cv2.resize(img_cell, (cell_img_size, cell_img_size))
            else:
                img_cell = cv2.resize(img_cell, current_shape)
        elif cell_img_size is not None:
            img_cell = cv2.resize(img_cell, (cell_img_size, cell_img_size))

    elif cell_img_size is not None:
        img_cell = cv2.resize(img_cell, (cell_img_size, cell_img_size))
    return img_cell


def get_cell_copied(cell_img, augmentations=[], height=1024, width=1024):
    cell_img = cell_img/255.
    cell_height, cell_width = cell_img.shape[:2]
    cell_img_tiled = np.tile(cell_img, [height // cell_height + 1, width // cell_width + 1, 1])
    cell_img_tiled = cell_img_tiled[:height, :width, :]
    images = [cell_img_tiled]

    for aug_func in augmentations:
        augmented = aug_func(image=cell_img)
        cell_img_aug = augmented['image']
        cell_img_aug_tiled = np.tile(cell_img_aug, [height // cell_height + 1, width // cell_width + 1, 1])
        cell_img_aug_tiled = cell_img_aug_tiled[:height, :width, :]
        images.append(cell_img_aug_tiled)

    return images


def open_rgb(image_id,
             folder_root='input/hpa-single-cell-image-classification/train'):  # a function that reads RGB image
    colors = ['red', 'green', 'blue']
    img = [cv2.imread(f'{folder_root}/{image_id}_{color}.png', cv2.IMREAD_GRAYSCALE)
           for color in colors]
    img = np.stack(img, axis=-1)
    return img


def open_rgby(image_id,
              folder_root='input/hpa-single-cell-image-classification/train'):  # a function that reads RGBY image
    colors = ['red', 'green', 'blue', 'yellow']
    img = [cv2.imread(f'{folder_root}/{image_id}_{color}.png', cv2.IMREAD_GRAYSCALE)
           for color in colors]
    img = np.stack(img, axis=-1)
    return img


def get_new_class_name_indices_in_prev_comp_data():
    class_names = [class_name.split('. ')[1].strip() for class_name in SPECIFIED_CLASS_NAMES.split('\n')]

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

    old_comp_class_names = [class_name.split('. ')[1].strip() for class_name in
                            old_comp_specified_class_names.split('\n')]

    new_name_index_2_old_name_index = dict()
    for new_class_index, class_name_new in enumerate(class_names):
        if class_name_new in old_comp_class_names:
            new_name_index_2_old_name_index[new_class_index] = old_comp_class_names.index(class_name_new)
    return list(new_name_index_2_old_name_index.values())


def get_cell_img_with_mask(img_id, cell_i, is_public_data, return_mask=True, target_img_size=1024):
    img_rgby = open_rgby(img_id,
                         folder_root='input/publichpa_1024/' if is_public_data else 'input/hpa-single-cell-image-classification/train')
    scale_factor = target_img_size / img_rgby.shape[0]
    bboxes_path_root = 'input/cell_bboxes_public' if is_public_data else 'input/cell_bboxes_train'
    bboxes_path = os.path.join(bboxes_path_root, f'{img_id}.pkl')
    bboxes_df = pd.read_pickle(bboxes_path)

    cell_bbox = bboxes_df.loc[cell_i + 1]

    img_ = img_rgby[cell_bbox['y_min']:cell_bbox['y_max'], cell_bbox['x_min']:cell_bbox['x_max'], :]
    img_[cell_bbox['cell_rows_del'], cell_bbox['cell_cols_del'], :] = 0
    required_shape = (int(img_.shape[1] * scale_factor), int(img_.shape[0] * scale_factor))
    img_ = cv2.resize(img_, required_shape)
    if not return_mask:
        return img_

    cell_mask = img_.mean(axis=-1) > 5
    return img_, cell_mask.astype('uint8')
