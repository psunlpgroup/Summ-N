import os.path
import json
from tqdm import tqdm
import multiprocessing
from typing import Dict, List, Tuple

from models.data_segment.segmentor_core import TargetSplitterCore
from utils.dataset import *
from utils.tools import add_sys_path

add_sys_path() # for ROUGE to find the root path

class TargetSegmentor(object):
    def __init__(self, cfg, data: Dict[str, List] = None, labels: Dict[str, List] = None, load_from_file=True):
        self.cur_stage = cfg.cur_stage
        self.cfg = getattr(cfg, f"stage{self.cur_stage}")
        self.output_path = cfg.train.output_path
        self.splitter = TargetSplitterCore()

        # load source and targets
        if load_from_file is True:
            for data_type in ['train', 'val', 'test']:
                # read data from the output of previous stage
                stage_path = os.path.join(self.output_path, f"stage_{self.cur_stage}")

                data_path = os.path.join(stage_path, f"{data_type}.source")
                self.data = read_list_asline(data_path)
                label_path = os.path.join(stage_path, f"{data_type}_duplicated.target")
                self.labels = read_list_asline(label_path)
        else:
            self.data = data
            self.labels = labels

        # store segmented targets, and the unordered segment
        self.target = {'train': [], 'val': [], 'test': []}
        self.best_label_with_scores = {'train': [], 'val': [], 'test': []}

    def segment(self) -> Tuple[Dict[str, list],Dict[str, list]]:
        for data_type in ['train', 'val', 'test']:
            # we use multiprocessing to accelerate the split process
            tasks = list(zip(self.data[data_type], self.labels[data_type], range(len(self.data[data_type]))))
            cores = min(multiprocessing.cpu_count(), self.cfg.cores_used)
            pool = multiprocessing.Pool(processes=cores)
            for i, (new_sents, new_tar) in tqdm(enumerate(pool.starmap(self.splitter.fast_rouge, tasks))):
                self.target[data_type].append(new_tar.strip())
                self.best_label_with_scores[data_type].append(new_sents)

        return self.target, self.best_label_with_scores

    def save(self):
        stage_path = os.path.join(self.output_path, f"stage_{self.cur_stage}")
        if not os.path.exists(stage_path):
            os.makedirs(stage_path)

        for data_type in ['train', 'val', 'test']:
            target_output_path = os.path.join(stage_path, f"{data_type}.target")
            best_label_path = os.path.join(stage_path, f"{data_type}_fastRouge.json")

            write_list_asline(target_output_path, self.target[data_type])
            with open(best_label_path, 'w', encoding='utf-8') as file:
                json.dump(self.best_label_with_scores[data_type], file)

