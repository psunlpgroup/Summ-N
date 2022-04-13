"""
Order of running

Stage 0 (Preprocessing)
./dataset_loader/{data name}

Stage 1 (coarse)
source_segment.py -> split source in stage 0
target_matching.py -> split target in stage 0
train_summerizer.sh -> training a summarizer
inference.py -> run multiple instance to generate summaries, and combine them
coarse_seg.py -> combine the generated summaries to form the input to the next stage

Stage 2 (fine-grained)
train_summarizer.sh -> train a fine-grained summarizer
inference.py -> run multiple instance to generate summaries, and combine them
AnyROUGE.py -> evaluate the final results

"""
import os

from argparse_dataclass import ArgumentParser

from utils.training_args import TrainArgs
from utils.configue import Configure
from utils.tools import get_dataloader
from utils.dataset import assertion_statis, write_finegrained_dataset, load_split_aslist
from utils.AnyROUGE import rouge

from models.data_segment.source_segment import SourceSegmentor
from models.data_segment.target_matching import TargetSegmentor
from models.gen_summary.inference import SummaryGenerator
from models.gen_summary.coarse_seg import CoarseSegCombiner

if __name__ == '__main__':
    # Parse all arguments
    parser = ArgumentParser(TrainArgs)
    training_args = parser.parse_args()
    args = Configure.Get(training_args.cfg)
    args.train = training_args # combine shell & file configs
    args.cur_stage = 0 # we name the data-collecting stage as stage_0

    # Load dataset using dataset loader
    dataset_loader = get_dataloader(args.dataset.loader_name)(args)
    source_data, target_data = dataset_loader.load()
    assertion_statis(source_data, target_data, f"Finish loading stage {args.cur_stage} dataset!")
    if args.train.save_intermediate is True:
        dataset_loader.save()

    # Coarse stages
    # args.cur_stage = args.dataset.stage_num - 1
    # if you want to skip coarse stages (or any stage) and the dataset for fine-grained stage is ready,
    # you can uncomment this line of code
    while args.cur_stage < args.dataset.stage_num - 1:
        args.cur_stage += 1

        # Source Segmentation
        source_segmentor = SourceSegmentor(args, source_data, target_data, load_from_file=False)
        split_source, duplicated_target, counter = source_segmentor.segment()
        source_segmentor.save()

        # Target Matching
        print(f"Start target matching of Stage {args.cur_stage}. This may take several minutes.")
        target_segmentor = TargetSegmentor(args, split_source, duplicated_target, load_from_file=False)
        target, _ = target_segmentor.segment()
        target_segmentor.save()

        assertion_statis(split_source, target, f"Finish loading stage {args.cur_stage} dataset!")

        if args.mode == "train":
            # Use collected data to run model training
            data_folder = os.path.join(args.train.output_path, f"stage_{args.cur_stage}")
            trainer_output_folder = os.path.join(data_folder, "trainer_output")
            os.system(f"bash models/train_summarizor.sh {data_folder} {trainer_output_folder} {args.train.cuda_devices}")

            # Inference using the trained checkpoint
            stage_arg = getattr(args, f"stage{args.cur_stage}")
            stage_arg.trainer_output_folder = trainer_output_folder
            summary_generator = SummaryGenerator(args, split_source, fine_grained=False)
            split_hypo = summary_generator.inference(bsz=8)
            if args.train.save_intermediate is True:
                summary_generator.save()

        else:
            # Inference using the trained checkpoint
            stage_arg = getattr(args, f"stage{args.cur_stage}")
            stage_arg.trainer_output_folder = os.path.join(args.checkpoint_dir, f"stage{args.cur_stage}")
            summary_generator = SummaryGenerator(args, split_source, fine_grained=False, test_mode=True)
            split_hypo = summary_generator.inference(bsz=8)
            if args.train.save_intermediate is True:
                summary_generator.save()

        # # Combine coarse segments to form the next stage's input
        combiner = CoarseSegCombiner(args, split_hypo, counter, load_from_file=False)
        # combiner = CoarseSegCombiner(args, None, counter, load_from_file=True)
        hypo = combiner.combine()
        combiner.save()

    # # Fine-grained Stage
    source_path = os.path.join(args.train.output_path, f"stage_{args.cur_stage}")
    cur_source = load_split_aslist(source_path, suffix='hypo')
    cur_target = target_data

    args.cur_stage += 1
    data_folder = os.path.join(args.train.output_path, f"stage_{args.cur_stage}")
    write_finegrained_dataset(cur_source, cur_target, data_folder)
    assertion_statis(cur_source, cur_target, f"Finish loading stage {args.cur_stage} dataset!")

    if args.mode == "train":
        trainer_output_folder = os.path.join(data_folder, "trainer_output")
        os.system(f"bash models/train_summarizor.sh {data_folder} {trainer_output_folder} {args.train.cuda_devices}")

        # Inference using the trained checkpoint
        stage_arg = getattr(args, f"stage{args.cur_stage}")
        stage_arg.trainer_output_folder = trainer_output_folder
        summary_generator = SummaryGenerator(args, cur_source, fine_grained=True)
        cur_hypo = summary_generator.inference()
        summary_generator.save()
    else:

        # Inference using the trained checkpoint
        stage_arg = getattr(args, f"stage{args.cur_stage}")
        stage_arg.trainer_output_folder = os.path.join(args.checkpoint_dir, f"stage{args.cur_stage}")
        summary_generator = SummaryGenerator(args, cur_source, fine_grained=True, test_mode=True)
        cur_hypo = summary_generator.inference()
        summary_generator.save()

    rouge_folder = os.path.join(data_folder, 'rouge_log') + '/'
    rouge_scores = rouge(cur_target['test'], cur_hypo['test'], rouge_folder)
