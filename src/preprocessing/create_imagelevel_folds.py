import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.simplefilter("ignore")
from sklearn.model_selection import StratifiedKFold

from ..data.utils import get_public_df_ohe, get_train_df_ohe, get_class_names

with open('../input/negs_with_staining_public.pkl', 'rb') as f:
    negs_with_staining_pub = pickle.load(f)
with open('../input/negs_with_staining.pkl', 'rb') as f:
    negs_with_staining = pickle.load(f)

train_df = get_train_df_ohe()
train_df = train_df[(train_df['Negative'] == 0) | train_df['img_base_path'].isin(negs_with_staining)]
img_paths_train = list(train_df['img_base_path'].values)
basepath_2_ohe_vector = {img:vec for img, vec in zip(train_df['img_base_path'], train_df.iloc[:, 3:].values)}


public_hpa_df_17 = get_public_df_ohe()
public_hpa_df_17 = public_hpa_df_17[(public_hpa_df_17['Negative'] == 0) | public_hpa_df_17['img_base_path'].isin(negs_with_staining_pub)]

public_basepath_2_ohe_vector = {img_path:vec for img_path, vec in zip(public_hpa_df_17['img_base_path'],
                                                                      public_hpa_df_17.iloc[:, 4:].values)}


basepath_2_ohe_vector.update(public_basepath_2_ohe_vector)


class_names = get_class_names()
all_labels = np.array(list(basepath_2_ohe_vector.values()))
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
class_counts = all_labels.sum(axis=0)
class_counts = pd.DataFrame({'Label': class_names, 'Number of Images': class_counts})
g = sns.barplot(x="Label", y="Number of Images",
                data=class_counts)
for item in g.get_xticklabels():
    item.set_rotation(90)
    item.set_fontsize(12)
g.set_ylim((0, 9000))
g.set_title('Main Train Images', fontsize=24)
g.set_xlabel('Class', fontsize=17)
g.set_ylabel('Number of Images', fontsize=17)
plt.savefig('../output/image_level_labels.png')

# splitting
label_combinations = []

for ohe_targets in basepath_2_ohe_vector.values():
    label_combinations.append(','.join(map(str, ohe_targets)))

label_combinations_counts = pd.Series(label_combinations).value_counts()
unique_label_combs = label_combinations_counts.index[(label_combinations_counts == 1).values]


img_paths_with_nonfreq_labels = []
img_paths_to_split_into_folds = []
y_to_split = []
for img_path, label_comb in zip(basepath_2_ohe_vector.keys(), label_combinations):
    if label_combinations_counts[label_comb] >= 3:
        img_paths_to_split_into_folds.append(img_path)
        y_to_split.append(label_comb)
    else:
        img_paths_with_nonfreq_labels.append(img_path)


kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1905)
folds = []
for trn_indices, val_indices in kf.split(img_paths_to_split_into_folds, y=y_to_split):
    trn_paths = [img_paths_to_split_into_folds[i] for i in trn_indices]
    val_paths = [img_paths_to_split_into_folds[i] for i in val_indices]
    folds.append([trn_paths, val_paths])

with open('../input/imagelevel_folds.pkl', 'wb') as f:
    pickle.dump(folds, f)