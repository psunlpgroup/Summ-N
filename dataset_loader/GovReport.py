import gzip
import os
import json
from  nltk import sent_tokenize

from utils.dataset import *
from dataset_loader.base_loader import LoaderBase


class Loader(LoaderBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.is_dialogue = False

    def load(self):
        for data_type in ['train', 'val', 'test']:
            source_path = os.path.join(self.cfg.train.dataset_path, f"{data_type}.source")
            target_path = os.path.join(self.cfg.train.dataset_path, f"{data_type}.target")
            source = read_list_asline(source_path)
            target = read_list_asline(target_path)
            self.data[data_type] = [sent_tokenize(x) for x in source] # we need to split document into sentences
            self.label[data_type] = target
        return self.data, self.label



