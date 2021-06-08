import pandas as pd
import os
from tqdm.auto import tqdm
import pickle

trn_cell_boxes_path='input/cell_bboxes_train'
public_cell_boxes_path='input/cell_bboxes_public'
labels_df = pd.read_hdf('output/image_level_labels.h5')
problematic_img_base_paths = []
for img_base_path in tqdm(set(labels_df.index.get_level_values(0))):
    is_from_train = 'train' in img_base_path
    cell_boxes_path = trn_cell_boxes_path if is_from_train else public_cell_boxes_path
    img_id = os.path.basename(img_base_path)
    bboxes_path = os.path.join(cell_boxes_path, f'{img_id}.pkl')
    bboxes_df = pd.read_pickle(bboxes_path)

    bboxes_cells_one_start = bboxes_df.index.values
    bboxes_cells = set(bboxes_cells_one_start - 1)
    labels_cells = set(labels_df.loc[img_base_path].index.values)
    iou = len(bboxes_cells.intersection(labels_cells))/len(bboxes_cells.union(labels_cells))
    if iou < 1:
        problematic_img_base_paths.append(img_base_path)
        print(len(problematic_img_base_paths))

with open('output/bbox_pred_inconsistent_basepaths.pkl', 'wb') as f:
    pickle.dump(problematic_img_base_paths, f)

"""REPAIR AFTER CASTING NEW PREDICTIONS STORED INTO image_level_labels_repaired.h5
import pandas as pd
df_repaired = pd.read_hdf('image_level_labels_repaired.h5')
repaired_img_basepaths = set(df_repaired['img_basepath'].values)
df_all = pd.read_hdf('image_level_labels.h5')
df_all_ = df_all.drop(repaired_img_basepaths)
df_repaired.set_index(['img_basepath', 'img_cell_number'], inplace=True)
df_all_repaired = pd.concat((df_all_, df_repaired))
df_all_repaired.to_hdf('image_level_labels.h5', key='data')
"""
