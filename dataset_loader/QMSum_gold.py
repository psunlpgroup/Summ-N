import gzip
import os
import json

from SummN.utils.dataset import *
from SummN.dataset_loader.base_loader import LoaderBase


class Loader(LoaderBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.is_dialogue = True
        self.has_query = True
        self.query = {'train':[], 'val':[],'test':[]}

    def load(self):
        raise NotImplementedError()
        for data_type in ['train', 'val', 'test']:
            data_path = os.path.join(self.cfg.train.dataset_path, f"jsonl/{data_type}.jsonl")

            data = []
            with open(data_path, 'r', encoding='utf-8') as file:
                for line in file:
                    data.append(json.loads(line))

            for sample in data:
                trans = [x['speaker'] + ' : ' + x['content'] for x in sample['meeting_transcripts']]

                all_query = []
                all_target = []
                for pair in sample['general_query_list']:
                    all_query.append(pair['query'])
                    all_target.append(pair['answer'])
                for pair in sample['specific_query_list']:
                    all_query.append(pair['query'])
                    all_target.append(pair['answer'])

                for q, t in zip(all_query, all_target):
                    self.query[data_type].append(q)
                    self.data[data_type].append(trans)
                    self.label[data_type].append(t)

        return self.data, self.label, self.query



