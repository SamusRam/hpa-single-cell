# credits: https://www.kaggle.com/rai555/hpa-duplicate-images-in-train

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from imagededup.methods import PHash
import logging



logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(
    f'Duplicates search'
)



# Helper functions
def convert_dict_to_df(duplicates_dict):
    duplicates_list = []
    scores = []

    for key, dup_items in duplicates_dict.items():
        image1 = key.split('_')[0]

        for item in dup_items:
            image2 = item[0].split('_')[0]
            if image1!=image2 and ((image1, image2) not in duplicates_list) and ((image2, image1) not in duplicates_list):
                    duplicates_list.append((image1, image2))
                    scores.append(item[1])

    duplicates_df = pd.DataFrame(duplicates_list, columns=['image1', 'image2'])
    duplicates_df['score'] = scores
    return duplicates_df


# The max_distance_threshold parameter of phash.find_duplicates() specifies the hamming distance below which retrieved duplicates are considered valid.  We'll start with a max_distance_threshold of 8. 

# In[5]:


phash = PHash()

encodings = phash.encode_images(image_dir='../input/hpa-single-cell-image-classification/train')
encodings_public = phash.encode_images(image_dir='../input/publichpa_1024')
encodings.update(encodings_public)

duplicates = phash.find_duplicates(encoding_map = encodings, scores = True, max_distance_threshold = 8)


duplicates_df = convert_dict_to_df(duplicates)
duplicates_df.to_csv('../input/duplicates.csv', index=False)