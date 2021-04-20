import sys
sys.path.insert(0, '..')
import argparse
import pickle
import pandas as pd
import torch
import torch.optim
from torch.backends import cudnn
import torch.nn.functional as F
from albumentations import Compose, VerticalFlip, HorizontalFlip, Rotate

from ..models.layers_bestfitting.loss import *
from tqdm.auto import tqdm
from ..models.networks_bestfitting.imageclsnet import init_network
from ..data.utils import get_train_df_ohe, get_public_df_ohe, get_cells_from_img, get_cell_copied
import multiprocessing


parser = argparse.ArgumentParser(description='PyTorch Protein Classification')
parser.add_argument('--model-folds-dir', default='../output/models/densenet121_1024_all_data__obvious_neg__gradaccum_20__start_lr_3e6', type=str, help='destination where trained network should be saved')
parser.add_argument('--gpu-id', default='0', type=str, help='gpu id used for training (default: 0)')
parser.add_argument('--arch', default='class_densenet121_large_dropout', type=str,
                    help='model architecture (default: class_densenet121_large_dropout)')
parser.add_argument('--num_classes', default=19, type=int, help='number of classes (default: 19)')
parser.add_argument('--in_channels', default=4, type=int, help='in channels (default: 4)')
parser.add_argument('--img_size', default=1024, type=int, help='image size (default: 512)')
parser.add_argument('--batch_size', default=8, type=int, help='train mini-batch size (default: 32)')
parser.add_argument('--workers', default=multiprocessing.cpu_count() - 1, type=int, help='number of data loading workers (default: 3)')
parser.add_argument('--num-folds', default=5, type=int)
parser.add_argument('--output-path', default='../output/image_level_labels_repaired_REPLACE.h5')
parser.add_argument('--fold-single', default=None, type=int)


def main():
    args = parser.parse_args()

    model_dir = args.model_folds_dir
    num_folds = args.num_folds

    fold_single = args.fold_single

    batch_size = args.batch_size

    # set cuda visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    cudnn.benchmark = True
    # cudnn.enabled = False

    # set random seeds
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    model_params = {}
    model_params['architecture'] = args.arch
    model_params['num_classes'] = args.num_classes
    model_params['in_channels'] = args.in_channels
    if 'efficientnet' in args.arch:
        model_params['image_size'] = args.img_size
        model_params['encoder'] = args.effnet_encoder

    models = []
    if fold_single is not None:
        for _ in range(fold_single):
            models.append('a')

    folds_list = [fold_single] if fold_single is not None else list(range(num_folds))
    for fold in folds_list:
        path = os.path.join(f'{model_dir}', f'fold{fold}', 'final.pth')
        final_checkpoint = torch.load(path)
        model_ = init_network(model_params)
        model_.load_state_dict(final_checkpoint['state_dict'])
        model_.cuda()
        model_.eval()
        models.append(model_)

    with open('../input/imagelevel_folds_obvious_staining_5.pkl', 'rb') as f:
        folds = pickle.load(f)

    img_basepaths_list = []
    img_cell_num_list = []
    label_probs_list = []

    vert_flip = VerticalFlip(always_apply=True)
    hor_flip = HorizontalFlip(always_apply=True)
    rot = Rotate(always_apply=True, limit=(89, 91))

    # with open('../output/bbox_pred_inconsistent_basepaths.pkl', 'rb') as f:
    #     img_ids_problematic = set([os.path.basename(x) for x in pickle.load(f)])
    #
    # print('img_ids_problematic', img_ids_problematic)

    for fold in folds_list:
        _, val_img_paths = folds[fold]
        # val_img_paths = ['../input/hpa-single-cell-image-classification/train/0032a07e-bba9-11e8-b2ba-ac1f6b6435d0']

        # val_img_paths = [x for x in val_img_paths if os.path.basename(x) in img_ids_problematic]
        #
        # print('val_img_paths', val_img_paths)

        train_df = get_train_df_ohe(clean_from_duplicates=True)
        public_df = get_public_df_ohe(clean_from_duplicates=True)

        available_paths = set(np.concatenate((train_df['img_base_path'].values, public_df['img_base_path'].values)))
        fold_img_paths = [path for path in val_img_paths if path in available_paths]

        for base_path in tqdm(fold_img_paths, desc=f'Processing fold {fold}'):
            cell_2_predictions_list = []
            # cell_imgs = get_cells_from_img(base_path, return_raw=True, target_img_size=1024)
            cell_images_tiled_all = []

            # cell_i = 0
            # for cell_img in cell_imgs:
            for cell_img in get_cells_from_img(base_path, return_raw=True, target_img_size=1024):
                cell_images_tiled_all.extend(get_cell_copied(cell_img, augmentations=[vert_flip, hor_flip, rot]))
                # import matplotlib.pyplot as plt
                # if cell_i  + 1 == 14:
                # for i in range(4, 5):
                #     plt.figure(figsize=(10, 10))
                #     plt.imshow(cell_images_tiled_all[-i][:, :, :3])
                #     plt.savefig(f'{cell_i}_{i}.png')
                # cell_i += 1


            for batch_i in range(len(cell_images_tiled_all)//batch_size + (1 if len(cell_images_tiled_all)%batch_size != 0 else 0)):
                classifier_batch_next = cell_images_tiled_all[batch_i*batch_size: (batch_i + 1)*batch_size]
                images_batch_torch_np = np.stack(classifier_batch_next).astype(np.float32)
                images_batch_torch_np = images_batch_torch_np.transpose((0, 3, 1, 2))
                with torch.no_grad():
                    cell_predictions_batch = F.sigmoid(models[fold](torch.from_numpy(images_batch_torch_np).cuda())).detach().cpu().numpy()
                #                     print(cell_predictions_batch.shape)
                cell_predictions_batch_per_cell = np.empty(
                    (cell_predictions_batch.shape[0] // 4, cell_predictions_batch.shape[1]))
                for row_i in range(cell_predictions_batch_per_cell.shape[0]):
                    cell_predictions_batch_per_cell[row_i] = cell_predictions_batch[row_i * 4: (row_i + 1) * 4].mean(
                        axis=0)
                cell_2_predictions_list.append(cell_predictions_batch_per_cell)

            if len(cell_2_predictions_list) == 0: continue
            cell_2_predictions_np = np.concatenate(cell_2_predictions_list) if len(cell_2_predictions_list) > 1 else cell_2_predictions_list[0]

            img_basepaths_list.extend([base_path for _ in range(len(cell_2_predictions_np))])
            img_cell_num_list.extend(list(range(len(cell_2_predictions_np))))
            label_probs_list.extend([pred_vec for pred_vec in cell_2_predictions_np])
            # print(cell_2_predictions_np.shape, cell_2_predictions_np[13][15], cell_2_predictions_np[14][15])

    image_level_labels_df = pd.DataFrame({'img_basepath': img_basepaths_list,
                                          'img_cell_number': img_cell_num_list,
                                          'image_level_pred': label_probs_list})
    output_path = args.output_path.replace('REPLACE', f'{fold_single}') if fold_single is not None else args.output_path.replace('_REPLACE', '')
    image_level_labels_df.to_hdf(output_path, key='data')


if __name__ == '__main__':
    print('%s: calling main function ... \n' % os.path.basename(__file__))
    main()
    print('\nsuccess!')
