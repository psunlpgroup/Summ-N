import json
import os
from typing import Dict, List, Tuple

from utils.dataset import *
from models.data_segment.segmentor_core import SourceSplitterCore


class SourceSegmentor(object):
    def __init__(self, cfg, data: Dict[str,List] = None, labels: Dict[str, List] = None, load_from_file=True):
        self.cur_stage = cfg.cur_stage
        self.cfg = getattr(cfg, f"stage{self.cur_stage}")
        self.output_path = cfg.train.output_path
        self.splitter = SourceSplitterCore(self.cfg.input_max_token - 34) # 34 tokens for question, if any

        # load source and targets
        if load_from_file is True:
            for data_type in ['train','val', 'test']:
                # we use the coarse summary as the intput to the next stage, except for stage 0
                if self.cur_stage == 1:
                    suffix = 'source'
                else:
                    suffix = 'hypo'

                # read data from the output of previous stage
                stage_path = os.path.join(self.output_path, f"stage_{self.cur_stage - 1}")

                data_path = os.path.join(stage_path, f"{data_type}.{suffix}")
                self.data = read_list_asline(data_path)
                # we always pick the target from stage 0 as the input target
                label_path = os.path.join(self.output_path, "stage_0", f"{data_type}.target")
                self.labels = read_list_asline(label_path)
        else:
            self.data = data
            self.labels = labels

        # store segmented source and targets
        self.split_source = {'train':[], 'val':[], 'test':[]}
        self.dupli_target = {'train':[], 'val':[], 'test':[]}
        self.count = {'train':[], 'val':[], 'test':[]}

    def segment(self, query=None) -> Tuple[Dict[str, list], Dict[str, list], Dict[str, List[int]]]:
        for data_type in ['train', 'val', 'test']:
            # source segmentation, target duplication, record the counting file
            for i, (trans, target) in enumerate(zip(self.data[data_type], self.labels[data_type])):
                split_trans = self.splitter.segment_one_sample(trans)
                split_trans = [' '.join(x) for x in split_trans]
                if query is not None:
                    split_trans = [query[i]] + split_trans

                for tran in split_trans:
                    self.split_source[data_type].append(tran.strip()+'\n')
                    self.dupli_target[data_type].append(target.strip()+'\n')

                self.count[data_type].append(len(split_trans))
        return self.split_source, self.dupli_target, self.count

    def save(self):
        stage_path = os.path.join(self.output_path, f"stage_{self.cur_stage}")
        if not os.path.exists(stage_path):
            os.makedirs(stage_path)

        for data_type in ['train', 'val', 'test']:
            source_output_path = os.path.join(stage_path, f"{data_type}.source")
            target_output_path = os.path.join(stage_path, f"{data_type}_duplicated.target")
            count_output_path = os.path.join(stage_path, f"{data_type}_count.json")

            write_list_asline(source_output_path, self.split_source[data_type])
            write_list_asline(target_output_path, self.dupli_target[data_type])

            with open(count_output_path, 'w', encoding='utf-8') as file:
                json.dump(self.count[data_type], file)