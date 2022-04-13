import json
import os
from typing import List,Dict
from utils.dataset import read_list_asline, write_list_asline

class CoarseSegCombiner(object):
    def __init__(self, cfg,
                 split_results: Dict[str, List] = None,
                 count_list: Dict[str, List[int]] = None,
                 # origin_target: Dict[str, List] = None,
                 load_from_file = False):
        self.cfg = cfg
        self.stage_folder = os.path.join(self.cfg.train.output_path, f"stage_{self.cfg.cur_stage}")

        if load_from_file:
            self.split_results = {'train':[], 'test':[], 'val':[]}
            self.count_list = {'train':[], 'test':[], 'val':[]}
            # self.target = {'train':[], 'test':[], 'val':[]}
            for data_type in ['train','val','test']:
                data_path = os.path.join(self.stage_folder, f"{data_type}_split.hypo")
                self.split_results[data_type] = read_list_asline(data_path)
                count_path = os.path.join(self.stage_folder, f"{data_type}_count.json")
                self.count_list[data_type] = json.load(open(count_path))
                # target_path = os.path.join(self.cfg.train.output_path, f"stage_0", f"{data_type}.target")
                # self.target[data_type] = read_list_asline(target_path)

        else:
            self.split_results = split_results
            self.count_list = count_list
            # self.target = origin_target

        # store combined hypos
        self. combined_hypos = {'train':[], 'test':[], 'val':[]}

    def combine(self):
        for data_type in ['train', 'val', 'test']:
            start = 0
            for lengths in self.count_list[data_type]:
                if start + lengths <= len(self.split_results[data_type]):
                    end = start + lengths
                    self.combined_hypos[data_type].append("<s> " + " </s> ".join(self.split_results[data_type][start:end]))
                    start = start + lengths
                else:
                    break
        return self.combined_hypos

    def save(self):
        for data_type in ['train','val','test']:
            save_path = os.path.join(self.stage_folder, f"{data_type}.hypo")
            write_list_asline(save_path, self.combined_hypos[data_type])
