from src.commons.config.config_bestfitting import *
from .densenet import class_densenet121_dropout, class_densenet121_large_dropout
from .inception_v3 import class_inceptionv3_dropout
from .resnet import class_resnet34_dropout, class_resnet18_dropout
from .efficientnet import class_efficientnet_dropout

model_names = {
    'class_densenet121_dropout': 'external_crop512_focal_slov_hardlog_class_densenet121_dropout_i768_aug2_5folds/fold0/final.pth',
    'class_densenet121_large_dropout': 'external_crop1024_focal_slov_hardlog_clean_class_densenet121_large_dropout_i1536_aug2_5folds/fold0/final.pth',
    # 'class_inceptionv3_dropout': 'inception_v3_google-1a9a5a14.pth',
    # 'class_resnet34_dropout': 'resnet34-333f7ec4.pth',
    # 'class_resnet18_dropout': 'resnet18-5c106cde.pth',
}

def init_network(params):
    architecture = params.get('architecture', 'class_densenet121_dropout')

    if architecture in model_names:
        pretrained_file = os.path.join(PRETRAINED_DIR, model_names[architecture])
        params.update({'pretrained_file':pretrained_file})
        print(">> Using pre-trained model.")
    net = eval(architecture)(**params)
    return net
