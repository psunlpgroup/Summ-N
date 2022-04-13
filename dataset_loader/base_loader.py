import os
from typing import List, Dict, Tuple

from utils.dataset import *

class LoaderBase(object):
    def __init__(self,cfg):
        self.data = {"train":[], "test":[], "val":[]}
        self.label = {"train":[], "test":[], "val":[]}
        self.cfg = cfg
        self.is_dialogue = True

    # load dataset and return a list of source and target
    def load(self) -> Tuple[Dict[str, list], Dict[str, list]]:
        raise NotImplementedError

    def save(self):
        for data_type in ['train', 'val', 'test']:
            # write to stage 0 path
            stage_path = os.path.join(self.cfg.train.output_path, f"stage_{self.cfg.cur_stage}")
            if not os.path.exists(stage_path):
                os.makedirs(stage_path)

            source_path = os.path.join(stage_path, f"{data_type}.source")
            target_path = os.path.join(stage_path, f"{data_type}.target")

             # join all turns in dialogue dataset
            write_list_asline(source_path, [' '.join(x) for x in self.data[data_type]])
            write_list_asline(target_path, self.label[data_type])
