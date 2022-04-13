from typing import Dict
from tqdm import tqdm
import os

import torch
from fairseq.models.bart import BARTModel
from utils.dataset import write_list_asline

class SummaryGenerator(object):
    def __init__(self, cfg, data: Dict[str, list], fine_grained=False, test_mode=False):
        self.cfg = cfg
        self.stage_cfg = getattr(cfg, f"stage{cfg.cur_stage}")
        self.data = data
        self.fine_grained = fine_grained
        self.test_mode = test_mode

        self.bart = BARTModel.from_pretrained(
            self.stage_cfg.trainer_output_folder,
            checkpoint_file='checkpoints/checkpoint_best.pt',
            data_name_or_path="./bin"
        )

        self.bart.cuda()
        self.bart.eval()
        self.bart.half()

        # Store generated samples
        self.hypo = {'train':[], 'val':[], 'test':[]}

    def inference(self, bsz=8) -> Dict[str, list]:
        for data_type in ['train', 'val', 'test']:

            # we only need the test results on the last stage
            if data_type != 'test' and (self.fine_grained or self.test_mode):
                continue

            # change the maximum input tokens per batch
            self.bart.cfg.dataset.batch_size_valid = bsz
            self.bart.cfg.dataset.max_tokens_valid = bsz * 1024

            count = 1
            slines = [self.data[data_type][0]]
            for sline in tqdm(self.data[data_type][1:]):
                if count % bsz == 0:
                    with torch.no_grad():
                        # parameters https://github.com/pytorch/fairseq/blob/master/fairseq/sequence_generator.py
                        hypotheses_batch = self.bart.sample(slines,
                                                            beam=self.stage_cfg.beam,
                                                            lenpen=self.stage_cfg.lenpen,
                                                            max_len_b=self.stage_cfg.max_len_b,
                                                            min_len=self.stage_cfg.min_len,
                                                            no_repeat_ngram_size=self.stage_cfg.no_repeat_ngram_size)
                    for hypothesis in hypotheses_batch:
                        self.hypo[data_type].append(hypothesis)
                    slines = []

                slines.append(sline.strip())
                count += 1

            if slines:
                hypotheses_batch = self.bart.sample(slines,
                                                    beam=self.stage_cfg.beam,
                                                    lenpen=self.stage_cfg.lenpen,
                                                    max_len_b=self.stage_cfg.max_len_b,
                                                    min_len=self.stage_cfg.min_len,
                                                    no_repeat_ngram_size=self.stage_cfg.no_repeat_ngram_size)
                for hypothesis in hypotheses_batch:
                    self.hypo[data_type].append(hypothesis)

        return self.hypo

    def save(self):
        for data_type in ['train', 'val', 'test']:

            # we only need the test results on the last stage
            if data_type != 'test' and (self.fine_grained or self.test_mode):
                continue

            data_folder = os.path.join(self.cfg.train.output_path, f"stage_{self.cfg.cur_stage}")
            save_path = os.path.join(data_folder, f"{data_type}_split.hypo" if not self.fine_grained
                                                  else f"{data_type}.hypo")
            write_list_asline(save_path, self.hypo[data_type])