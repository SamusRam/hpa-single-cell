import os
ope = os.path.exists
import numpy as np
import socket
import warnings
warnings.filterwarnings('ignore')

sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
hostname = socket.gethostname()
print('run on %s' % hostname)

PI  = np.pi
INF = np.inf
EPS = 1e-12

IMG_SIZE      = 1024
NUM_CLASSES   = 19
ID            = 'Id'
PREDICTED     = 'Predicted'
TARGET        = 'Target'
RESULT_DIR = '../output'
PRETRAINED_DIR = '../input/pretrained_models/models'


LABEL_NAMES = {
    0:  "Nucleoplasm",
    1:  "Nuclear membrane",
    2:  "Nucleoli",
    3:  "Nucleoli fibrillar center",
    4:  "Nuclear speckles",
    5:  "Nuclear bodies",
    6:  "Endoplasmic reticulum",
    7:  "Golgi apparatus",
    8:  "Intermediate filaments",
    9:  "Actin filaments",
    10:  "Microtubules",
    11:  "Mitotic spindle",
    12:  "Centrosome",
    13:  "Plasma membrane",
    14:  "Mitochondria",
    15:  "Aggresome",
    16:  "Cytosol",
    17:  "Vesicles and punctate cytosolic patterns",
    18:  "Negative"
}
LABEL_NAME_LIST = [LABEL_NAMES[idx] for idx in range(len(LABEL_NAMES))]

COLOR_INDEXS = {
    'red': 0,
    'green': 1,
    'blue': 2,
    'yellow': 0,
}
