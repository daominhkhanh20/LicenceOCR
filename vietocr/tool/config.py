import yaml
from vietocr.tool.utils import download_config
import itertools
import os
from collections import Counter

url_config = {
        'vgg_transformer':'vgg-transformer.yml',
        'resnet_transformer':'resnet_transformer.yml',
        'resnet_fpn_transformer':'resnet_fpn_transformer.yml',
        'vgg_seq2seq':'vgg-seq2seq.yml',
        'vgg_convseq2seq':'vgg_convseq2seq.yml',
        'vgg_decoderseq2seq':'vgg_decoderseq2seq.yml',
        'base':'base.yml',
        }

def parser_cnt(path):
    with open(path, 'r') as file:
        data = file.readlines()
    data = [value.split('\t')[1].strip() for value in data]
    data = [list(value) for value in list(set(data))]
    list_chars = itertools.chain(*data)
    result = dict(Counter(list_chars))
    sum_value = sum([1/value for value in result.values()])
    for key in result:
        result[key] = (1 / result[key]) / sum_value
    return {'weight': result}
    
class Cfg(dict):
    def __init__(self, config_dict):
        super(Cfg, self).__init__(**config_dict)
        self.__dict__ = self

    @staticmethod
    def load_config_from_file(fname):
        #base_config = download_config(url_config['base'])
        base_config = {}
        with open(fname, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        base_config.update(config)
        if os.path.isfile(base_config.get('train_annotation', None)):
            base_config.update(parser_cnt(base_config.get('train_annotation')))

        return Cfg(base_config)
                                                     
    @staticmethod
    def load_config_from_name(name):
        base_config = download_config(url_config['base'])
        config = download_config(url_config[name])

        base_config.update(config)
        return Cfg(base_config)

    def save(self, fname):
        with open(fname, 'w') as outfile:
            yaml.dump(dict(self), outfile, default_flow_style=False, allow_unicode=True)

