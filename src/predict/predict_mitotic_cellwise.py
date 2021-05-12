# coding: utf-8
import sys
sys.path.insert(0, '..')
import argparse
import shutil
import pickle

import torch
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.nn import DataParallel
from torch.backends import cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import average_precision_score
import pandas as pd

from ..data.augment_util_bestfitting import train_multi_augment2
from ..models.layers_bestfitting.loss import *
from ..models.layers_bestfitting.scheduler import *
from ..models.networks_bestfitting.imageclsnet import init_network
from ..data.datasets import ProteinDatasetCellSeparateLoading #ProteinDatasetCellLevel
from ..data.utils import get_train_df_ohe, get_public_df_ohe, get_class_names
from src.commons.utils import Logger
import multiprocessing
import time

loss_names = ['FocalSymmetricHardLogLoss', 'SoftFocalSymmetricHardLogLoss', 'FocalSymmetricLovaszHardLogLoss']

parser = argparse.ArgumentParser(description='PyTorch Protein Classification')
parser.add_argument('--out_dir', default='densenet121_1024_all_data_obvious_neg', type=str, help='destination where trained network should be saved')
parser.add_argument('--gpu-id', default='0', type=str, help='gpu id used for training (default: 0)')
parser.add_argument('--arch', default='class_densenet121_large_dropout', type=str,
                    help='model architecture (default: class_densenet121_large_dropout)')
parser.add_argument('--effnet-encoder', default='efficientnet-b0', type=str)

parser.add_argument('--num_classes', default=19, type=int, help='number of classes (default: 19)')
parser.add_argument('--in_channels', default=4, type=int, help='in channels (default: 4)')
parser.add_argument('--loss', default='SoftCEHardLogLoss', choices=loss_names, type=str,
                    help='loss function: ' + ' | '.join(loss_names) + ' (deafault: SoftCEHardLogLoss)')
parser.add_argument('--scheduler', default='Adam20WarmUP', type=str, help='scheduler name')
parser.add_argument('--scheduler-lr-multiplier', default=1.0, type=float, help='scheduler lr multiplier')
parser.add_argument('--scheduler-epoch-offset', default=0, type=int, help='epoch offset for the scheduler')
parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run (default: 55)')
parser.add_argument('--img_size', default=512, type=int, help='image size (default: 512)')
parser.add_argument('--batch_size', default=32, type=int, help='train mini-batch size (default: 32)')
parser.add_argument('--workers', default=multiprocessing.cpu_count() - 1, type=int, help='number of data loading workers (default: 3)')
parser.add_argument('--fold', default=0, type=int, help='index of fold (default: 0)')
parser.add_argument('--clipnorm', default=1, type=int, help='clip grad norm')
parser.add_argument('--resume', default=None, type=str, help='name of the latest checkpoint (default: None)')
parser.add_argument('--load-state-dict-path', default=None, type=str, help='path to .h5 file with a state-dict to load before training (default: None)')
parser.add_argument('--cell-level-labels-path', default='../output/densenet121_pred.h5', type=str)
parser.add_argument('--eval-at-start', action='store_true')
parser.add_argument('--image-level-labels', action='store_true')
parser.add_argument('--normalize', action='store_true')
parser.add_argument('--ignore-negative', action='store_true')
parser.add_argument('--gradient-accumulation-steps', default=4, type=int)
parser.add_argument('--target-raw-img-size', default=None, type=int)
parser.add_argument('--include-nn-mitotic', action='store_true')
parser.add_argument('--upsample-minorities', action='store_true')
parser.add_argument('--all-gpus', action='store_true')

def main():
    args = parser.parse_args()

    log_out_dir = os.path.join(RESULT_DIR, 'logs', args.out_dir, 'fold%d' % args.fold)
    if not os.path.exists(log_out_dir):
        os.makedirs(log_out_dir)
    log = Logger()
    log.open(os.path.join(log_out_dir, 'log.train.txt'), mode='a')

    model_out_dir = os.path.join(RESULT_DIR, 'models', args.out_dir, 'fold%d' % args.fold)
    log.write(">> Creating directory if it does not exist:\n>> '{}'\n".format(model_out_dir))
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)

    # set cuda visible device
    if not args.all_gpus:
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

    model = init_network(model_params)

    if args.load_state_dict_path is not None:
        if args.load_state_dict_path == 'use-img-level-densenet-ckpt':
            model_dir = '../output/models/densenet121_1024_all_data__obvious_neg__gradaccum_20__start_lr_3e6'
            pretrained_ckpt_path = os.path.join(f'{model_dir}', f'fold{args.fold}', 'final.pth')
        else:
            pretrained_ckpt_path = args.load_state_dict_path
        init_pretrained = torch.load(pretrained_ckpt_path)
        model.load_state_dict(init_pretrained['state_dict'])

    if args.all_gpus:
        model = DataParallel(model)
    model.cuda()

    # define loss function (criterion)
    try:
        criterion = eval(args.loss)().cuda()
    except:
        raise(RuntimeError("Loss {} not available!".format(args.loss)))

    start_epoch = 0
    best_loss = 1e5
    best_epoch = 0
    best_focal = float('inf')

    # define scheduler
    try:
        scheduler = eval(args.scheduler)(scheduler_lr_multiplier=args.scheduler_lr_multiplier,
                                         scheduler_epoch_offset=args.scheduler_epoch_offset)
    except:
        raise (RuntimeError("Scheduler {} not available!".format(args.scheduler)))
    optimizer = scheduler.schedule(model, start_epoch, args.epochs)[0]

    # optionally resume from a checkpoint
    if args.resume:
        args.resume = os.path.join(model_out_dir, args.resume)
        if os.path.isfile(args.resume):
            # load checkpoint weights and update model and optimizer
            log.write(">> Loading checkpoint:\n>> '{}'\n".format(args.resume))

            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_epoch = checkpoint['best_epoch']
            best_focal = checkpoint['best_map']
            model.load_state_dict(checkpoint['state_dict'])

            optimizer_fpath = args.resume.replace('.pth', '_optim.pth')
            if os.path.exists(optimizer_fpath):
                log.write(">> Loading checkpoint:\n>> '{}'\n".format(optimizer_fpath))
                optimizer.load_state_dict(torch.load(optimizer_fpath)['optimizer'])
            log.write(">>>> loaded checkpoint:\n>>>> '{}' (epoch {})\n".format(args.resume, checkpoint['epoch']))
        else:
            log.write(">> No checkpoint found at '{}'\n".format(args.resume))

    # Data loading code
    train_transform = train_multi_augment2

    with open('../input/imagelevel_folds_obvious_staining_5.pkl', 'rb') as f:
        folds = pickle.load(f)
    fold = args.fold
    trn_img_paths, val_img_paths = folds[fold]

    train_df = get_train_df_ohe(clean_from_duplicates=True)
    basepath_2_ohe_vector = {img: vec for img, vec in zip(train_df['img_base_path'], train_df.iloc[:, 2:].values)}

    public_hpa_df_17 = get_public_df_ohe(clean_from_duplicates=True)
    public_basepath_2_ohe_vector = {img_path: vec for img_path, vec in zip(public_hpa_df_17['img_base_path'],
                                                                           public_hpa_df_17.iloc[:, 2:].values)}
    basepath_2_ohe_vector.update(public_basepath_2_ohe_vector)

    available_paths = set(np.concatenate((train_df['img_base_path'].values, public_hpa_df_17['img_base_path'].values)))
    trn_img_paths = [path for path in trn_img_paths if path in available_paths]
    val_img_paths = [path for path in val_img_paths if path in available_paths]
    labels_df = pd.read_hdf(args.cell_level_labels_path)

    # modifying minor class labels
    cherrypicked_mitotic_spindle = pd.read_csv('../input/mitotic_cells_selection.csv')

    cherrypicked_mitotic_spindle_img_cell = set(
        cherrypicked_mitotic_spindle[['ID', 'cell_i']].apply(tuple, axis=1).values)

    cherrypicked_mitotic_spindle_img_cell = {(img, cell_i - 1) for img, cell_i in cherrypicked_mitotic_spindle_img_cell}

    class_names = get_class_names()
    mitotic_spindle_class_i = class_names.index('Mitotic spindle')

    cherrypicked_mitotic_spindle_based_on_nn = pd.read_csv('../input/mitotic_pos_nn_added.csv')
    cherrypicked_mitotic_spindle_img_cell.update(set(cherrypicked_mitotic_spindle_based_on_nn[['ID', 'cell_i']].apply(tuple, axis=1).values))
    print('len cherrypicked_mitotic_spindle_img_cell', len(cherrypicked_mitotic_spindle_img_cell))
    mitotic_bool_idx = labels_df.index.isin(cherrypicked_mitotic_spindle_img_cell)

    def modify_label(labels, idx, val):
        labels[idx] = val
        return labels

    labels_df.loc[mitotic_bool_idx, 'image_level_pred'] = labels_df.loc[
        mitotic_bool_idx, 'image_level_pred'].map(lambda x: modify_label(x, mitotic_spindle_class_i, 1))

    labels_df = labels_df.loc[mitotic_bool_idx]

    valid_dataset = ProteinDatasetCellSeparateLoading(val_img_paths,
                                            labels_df=labels_df,
                                            img_size=args.img_size,
                                            in_channels=args.in_channels,
                                                      basepath_2_ohe=basepath_2_ohe_vector,
                                                      normalize=args.normalize,
                                                      target_raw_img_size=args.target_raw_img_size)
    valid_loader = DataLoader(
        valid_dataset,
        sampler=SequentialSampler(valid_dataset),
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=True
    )

    predict_and_store(valid_loader, model, valid_dataset.img_ids_cell, mitotic_idx=mitotic_spindle_class_i,
                      ouput_path=f'../output/mitotic_pred_fold_{args.fold}.csv')


def predict_and_store(valid_loader, model, img_ids_cell, mitotic_idx, ouput_path):

    model.eval()

    probs_list = []
    labels_list = []

    for it, iter_data in enumerate(valid_loader, 0):
        images, labels, indices = iter_data
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())

        outputs = model(images)

        logits = outputs
        probs = F.sigmoid(logits)


        probs_list.append(probs.cpu().detach().numpy())
        labels_list.append(labels.cpu().detach().numpy())

    probs = np.vstack(probs_list)

    results_df = pd.DataFrame({'ID': [x[0] for x in img_ids_cell], 'cell_i': [x[1] for x in img_ids_cell],
                               'pred': [prob[mitotic_idx] for prob in probs]})
    results_df.to_csv(ouput_path, index=None)


if __name__ == '__main__':
    print('%s: calling main function ... \n' % os.path.basename(__file__))
    main()
    print('\nsuccess!')
