import json
from SummN.utils.dataset import *
from SummN.dataset_loader.base_loader import LoaderBase

class Loader(LoaderBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.is_dialogue = True

    def load(self):
        for data_type in ['train','val','test']:
            if data_type == "val": # name of tms is dev not val
                data_path = os.path.join(self.cfg.train.dataset_path, "fd_dev.json")
            else:
                data_path = os.path.join(self.cfg.train.dataset_path, f"fd_{data_type}.json")

            with open(data_path, 'r', encoding='utf-8') as file:
                dataset = [json.loads(line) for line in file]
                for sample in dataset:
                    tar = sample['Recap'][0]
                    sou = [x.replace("@@ ", '') for x in sample['Transcript']]
                    tar = tar.replace("@@ ", '').replace('\n', '')
                    self.data[data_type].append(sou) # source is a list of string for dialogue dataset
                    self.label[data_type].append(tar) # target is a string

        return self.data, self.label




