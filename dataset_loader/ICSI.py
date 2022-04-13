import gzip
import os
import json

from utils.dataset import *
from dataset_loader.base_loader import LoaderBase


class Loader(LoaderBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.is_dialogue = True

    def load(self):
        for data_type in ['train', 'val', 'test']:
            if data_type == "val":  # name of AMI is dev not val
                data_path = os.path.join(self.cfg.train.dataset_path, "dev")
            else:
                data_path = os.path.join(self.cfg.train.dataset_path, data_type)

            samples = []
            for gz_name in os.listdir(data_path):
                if 'gz' not in gz_name:
                    continue
                sample_path = os.path.join(data_path, gz_name)
                with gzip.open(sample_path, 'rb') as file:
                    for line in file:
                        samples.append(json.loads(line))

            for sample in samples:
                # get meetings & summary
                meeting = []
                for turn in sample['meeting']:
                    sent = turn['role'] + ' ' + turn['speaker'] + " : "
                    sent += ' '.join(turn['utt']['word'])
                    meeting.append(sent)
                summary = ' '.join(sample['summary'])

                self.data[data_type].append(meeting)
                self.label[data_type].append(summary)

        return self.data, self.label



