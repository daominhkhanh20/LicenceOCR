from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer
import torch

config = Cfg.load_config_from_name('vgg_transformer')
config['vocab'] = 'ABCDEFGHKLMNPSTUVXYZ0123456789-.'

dataset_params = {
    'name': 'hw',
    'data_root': './ocr/',
    'train_annotation': 'train.txt',
    'valid_annotation': 'val.txt'
}

params = {
    'print_every': 50,
    'valid_every': 1000,
    'iters': 10000,
    'checkpoint': '/checkpoint/transformerocr_checkpoint.pth',
    'export': './weights/transformerocr.pth',
    'metrics': 10000,
    'batch_size': 64,
    'lr': 1e-4
}

config['trainer'].update(params)
config['dataset'].update(dataset_params)
config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer = Trainer(config, pretrained=True)
trainer.train()