import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import BallTree, kneighbors_graph
from scipy.sparse.csgraph import laplacian
from petsc4py import PETSc
from slepc4py import SLEPc
from sporco.admm import bpdn
from tqdm.auto import tqdm
import os
from copy import deepcopy
import multiprocessing
import time
from sklearn.decomposition import PCA
import logging
from random import sample

from ..data.utils import get_train_df_ohe, get_public_df_ohe, get_class_names, get_masks_precomputed, open_rgb

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--num-cores", type=int, default=multiprocessing.cpu_count()//5)
parser.add_argument("--num-graph-neighbours", type=int, default=200)
parser.add_argument("--num-eigenvectors", type=int, default=50)
parser.add_argument("--output-path", default=None)

args = parser.parse_args()
FOLD_I = args.fold
NUM_CORES = args.num_cores
N_NEIGHBS = args.num_graph_neighbours
N_EIGENVECTORS = args.num_eigenvectors
OUTPUT_PATH = args.output_path
if OUTPUT_PATH is None:
    OUTPUT_PATH = f'../output/denoising_{FOLD_I}_{time.strftime("%Y%m%d_%H%M%S")}'
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(
    f'Graph De-noising: {os.path.basename(OUTPUT_PATH)}'
)

#############################
train_df = get_train_df_ohe()
img_paths_train = list(train_df['img_base_path'].values)
trn_basepath_2_ohe_vector = {img: vec for img, vec in zip(train_df['img_base_path'], train_df.iloc[:, 3:].values)}

public_hpa_df = get_public_df_ohe()
public_basepath_2_ohe_vector = {img_path: vec for img_path, vec in
                                zip(public_hpa_df['img_base_path'], public_hpa_df.iloc[:, 4:].values)}

trn_basepath_2_ohe_vector.update(public_basepath_2_ohe_vector)

# mappings ids
img_base_path_2_id = dict()
img_id_2_base_path = dict()
for img_base_path in trn_basepath_2_ohe_vector.keys():
    img_id = os.path.basename(img_base_path)
    img_base_path_2_id[img_base_path] = img_id
    img_id_2_base_path[img_id] = img_base_path

# (img id, mask_i) 2 emb path
img_id_2_emb_path = dict()

for root_path in ['../input/bestfitting_densenet_embs_train/', '../input/bestfitting_densenet_embs_public/']:
    trn_embs = [os.path.splitext(x)[0] for x in os.listdir(root_path)]

    for id_mask in trn_embs:
        img_id, mask_i = id_mask.split('__')
        img_id_2_emb_path[(img_id, int(mask_i))] = os.path.join(root_path, id_mask)


# folds
with open('../input/denoisining_folds.pkl', 'rb') as f:
    fold_2_imgId_2_maskIndices = pickle.load(f)

# ## fold encodings, labels, ids
encodings_global = []
weak_labels_global = []
img_id_mask_global = []

for img_id, mask_indices in tqdm(fold_2_imgId_2_maskIndices[FOLD_I].items(), desc='Gathering fold encodings and labels'):
    for mask_i in mask_indices:
        if (img_id, mask_i) in img_id_2_emb_path and os.path.exists(f'{img_id_2_emb_path[(img_id, mask_i)]}.npz'):
            emb = np.load(f'{img_id_2_emb_path[(img_id, mask_i)]}.npz')['arr_0']

            encodings_global.append(emb)
            img_labels_ohe = trn_basepath_2_ohe_vector[img_id_2_base_path[img_id]]
            weak_labels_global.append(img_labels_ohe)
            img_id_mask_global.append((img_id, mask_i))

# PCA embeddings
pca = PCA(n_components=2)
encodings_pca = pca.fit_transform(np.array(encodings_global))
labels_np = np.vstack(weak_labels_global)
single_class_bool_idx = labels_np.sum(axis=1) == 1
colors = plt.cm.tab20(np.arange(19))
class_names = get_class_names()
plt.figure(figsize=(9, 9))
for i, color, target_name in zip(range(19), colors, class_names):
    bool_idx = np.logical_and(single_class_bool_idx, labels_np[:, i] == 1)
    plt.scatter(encodings_pca[bool_idx, 0], encodings_pca[bool_idx, 1], color=color, alpha=.8,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.savefig(os.path.join(OUTPUT_PATH, 'pca_check.png'))

# Noise reduction
# KNN graph
ball_tree = BallTree(np.array(encodings_global))

knn_graph_200 = kneighbors_graph(ball_tree, n_neighbors=N_NEIGHBS, mode='connectivity', include_self=True,
                                 n_jobs=1)

knn_graph_200 = ((knn_graph_200 + knn_graph_200.T) > 0).astype(np.int)

with open(os.path.join(OUTPUT_PATH,f'knn_graph_{N_NEIGHBS}.pkl'), 'wb') as f:
    pickle.dump(knn_graph_200, f)

laplacian_normed = laplacian(knn_graph_200, normed=True)

laplacian_normed_csr = laplacian_normed.tocsr()
p1 = laplacian_normed_csr.indptr
p2 = laplacian_normed_csr.indices
p3 = laplacian_normed_csr.data
petsc_laplacian_normed_mat = PETSc.Mat().createAIJ(size=laplacian_normed_csr.shape, csr=(p1, p2, p3))


def solve_eigensystem(A, number_of_requested_eigenvectors, problem_type=SLEPc.EPS.ProblemType.HEP):
    # Create the result vectors
    xr, xi = A.createVecs()

    # Setup the eigensolver
    E = SLEPc.EPS().create()
    E.setOperators(A, None)
    E.setDimensions(number_of_requested_eigenvectors, PETSc.DECIDE)
    E.setProblemType(problem_type)
    E.setFromOptions()
    E.setWhichEigenpairs(E.Which.SMALLEST_REAL)

    # Solve the eigensystem
    E.solve()

    print("")
    its = E.getIterationNumber()
    print("Number of iterations of the method: %i" % its)
    sol_type = E.getType()
    print("Solution method: %s" % sol_type)
    nev, ncv, mpd = E.getDimensions()
    print("Number of requested eigenvalues: %i" % nev)
    tol, maxit = E.getTolerances()
    print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))
    nconv = E.getConverged()
    print("Number of converged eigenpairs: %d" % nconv)

    if nconv > 0:
        eigenvalues, eigenvectors = [], []
        for i in range(min(nconv, number_of_requested_eigenvectors)):
            k = E.getEigenpair(i, xr, xi)
            if k.imag != 0.0:
                print("oooooops")
            else:
                eigenvalues.append(k.real)
                eigenvectors.append(xr.array.copy())
    return eigenvalues, eigenvectors


eigenvalues, eigenvectors = solve_eigensystem(petsc_laplacian_normed_mat,
                                              number_of_requested_eigenvectors=N_EIGENVECTORS)

with open(os.path.join(OUTPUT_PATH, f'eigenvalues_{N_NEIGHBS}.pkl'), 'wb') as f:
    pickle.dump(eigenvalues, f)
with open(os.path.join(OUTPUT_PATH, f'eigenvectors_{N_NEIGHBS}.pkl'), 'wb') as f:
    pickle.dump(eigenvectors, f)

# utils


def soft_thr_matrices(x, y, gamma=0.12):
    z_1 = np.maximum(x - gamma, y)
    z_2 = np.maximum(0, np.minimum(x + gamma, y))
    f_1 = 0.5 * np.power(z_1 - x, 2) + gamma * np.absolute(z_1 - y)
    f_2 = 0.5 * np.power(z_2 - x, 2) + gamma * np.absolute(z_2 - y)
    return np.where(f_1 <= f_2, z_1, z_2)


def solve_for_label(j=0):
    Y_j = Y_opt[:, [j]]
    # Initialise and run BPDN object for best lmbda
    b = bpdn.BPDN(V_m, Y_j, lmbda, opt)
    A_j = b.solve()
    #     print("BPDN solve time: %.2fs" % b.timer.elapsed('solve'))
    return A_j


def get_current_A(Y_opt):
    A_list = []
    for j in range(Y_opt.shape[-1]):
        A_list.append(solve_for_label(j))
    return np.hstack(A_list)


def agrmax_superpixel_labels(Y):
    max_element = np.repeat(Y.max(1).reshape(-1, 1), Y.shape[1], axis=1)
    return ((Y == max_element) & (max_element > 0)).astype(np.int)


def thresholded_superpixel_labels(Y, threshold=0.5):
    return (Y >= threshold).astype(np.int)


def thresholded_superpixel_labels_adaptive(Y):
    max_element = np.repeat(Y.max(1).reshape(-1, 1), Y.shape[1], axis=1)
    return (Y > 0.9 * max_element).astype(np.int)

# Init denoising
l1_weights = np.array([eig ** 0.5 for eig in eigenvalues])
l1_weights = np.expand_dims(l1_weights, axis=-1)

Y_init = np.vstack(weak_labels_global)

# D -> V_m, Nxm
N = len(encodings_global)  # Number of cells
m = len(eigenvectors)  # Number of eigenvectors(dictionary size)

# Construct random dictionary and random sparse coefficients
V_m = np.array(eigenvectors).T
Y_opt = Y_init.copy()
lmbda = 0.01
opt = bpdn.BPDN.Options({'Verbose': False, 'MaxMainIter': 1000,
                         'RelStopTol': 1e-5, 'AutoRho': {'RsdlTarget': 1.0}, 'L1Weight': l1_weights, 'Verbose': False})

# Optimization

Y_opt_prev = None
iter_i = 0
while Y_opt_prev is None or np.isclose(Y_opt, Y_opt_prev, rtol=0.01).sum() / np.multiply(*Y_opt.shape) < 0.999:
    Y_opt_prev = deepcopy(Y_opt)
    A = get_current_A(Y_opt)
    F = V_m.dot(A)
    Y_opt = soft_thr_matrices(F, Y_init)
    iter_i += 1
    logger.info(f'Iteration {iter_i}: {np.isclose(Y_opt, Y_opt_prev, rtol=0.01).sum() / np.multiply(*Y_opt.shape):.5f}')

with open(os.path.join(OUTPUT_PATH, f'final_labels.pkl'), 'wb') as f:
    pickle.dump(Y_opt, f)

with open(os.path.join(OUTPUT_PATH, f'img_id_mask_i.pkl'), 'wb') as f:
    pickle.dump(img_id_mask_global, f)


# Visualizations
output_vis_hists = os.path.join(OUTPUT_PATH, 'histograms_final')
if not os.path.exists(output_vis_hists):
    os.makedirs(output_vis_hists)
for class_i, class_name in enumerate(class_names):
    plt.figure()
    plt.hist(Y_opt[:, class_i])
    plt.title(f'{class_name}, {Y_init[:, class_i].mean():.5f}, new: {sum(Y_opt[:, class_i] > 0.1)/len(Y_opt): .5f}')
    logger.info(f'For class {class_name} initial proportion was {Y_init[:, class_i].mean():.5f}, and new proportion is {sum(Y_opt[:, class_i] > 0.1)/len(Y_opt): .5f}.')
    plt.savefig(os.path.join(output_vis_hists, f'{class_name}.png'))


for class_i, class_name in enumerate(class_names):
    output_class_imgs_path = os.path.join(OUTPUT_PATH, f'results_visualizations_{class_name}')

    img_id_mask_global_removed = np.array(img_id_mask_global)[(Y_init[:, class_i] == 1) & (Y_opt[:, class_i] < 0.2)]
    if len(img_id_mask_global_removed):
        output_removed_path = os.path.join(output_class_imgs_path, 'removed_cases')
        if not os.path.exists(output_removed_path):
            os.makedirs(output_removed_path)

        for img_id, mask_i in sample(list(img_id_mask_global_removed), min(len(img_id_mask_global_removed), 30)):
            try:
                masks = get_masks_precomputed([f'{img_id}__{mask_i}'], '../input/hpa_cell_mask')[0]
            except:
                masks = get_masks_precomputed([f'{img_id}__{mask_i}'], '../input/hpa_cell_mask_public')[0]
            mask_bool = masks == mask_i
            img = open_rgb(img_id)
            img[np.logical_not(mask_bool)] = 0
            plt.figure(figsize=(15, 15))
            plt.imshow(img)
            plt.title(f'Removed for {class_name}', fontsize=25)
            plt.savefig(os.path.join(output_removed_path, f'{img_id}_{mask_i}.png'))

    # added imgs
    img_id_mask_global_added = np.array(img_id_mask_global)[(Y_init[:, class_i] == 0) & (Y_opt[:, class_i] > 0.8)]
    if len(img_id_mask_global_added):
        output_added_path = os.path.join(output_class_imgs_path, 'added_cases')
        if not os.path.exists(output_added_path):
            os.makedirs(output_added_path)

        for img_id, mask_i in sample(list(img_id_mask_global_added), min(len(img_id_mask_global_added), 30)):
            try:
                masks = get_masks_precomputed(f'{img_id}__{mask_i}', '../input/hpa_cell_mask/')[0]
            except:
                masks = get_masks_precomputed(f'{img_id}__{mask_i}', '../input/hpa_cell_mask_public/')[0]
            mask_bool = masks == mask_i
            img = open_rgb(img_id)
            img[np.logical_not(mask_bool)] = 0
            plt.figure(figsize=(15, 15))
            plt.imshow(img)
            plt.title(f'Added for {class_name}', fontsize=25)
            plt.savefig(os.path.join(output_added_path, f'{img_id}_{mask_i}.png'))
