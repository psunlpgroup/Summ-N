from dataclasses import dataclass, field


@dataclass
class TrainArgs(object):
    cfg: str = field(
        default="",
        metadata={"help": "The path from ./configure to store configure file."})

    dataset_path: str = field(
        default="/data/yfz5488/src/SummScreen/TVMegaSite/",
        metadata={"help": "The absolute path to the dataset folder."}
    )

    output_path: str = field(
        default="./output/SummScreenMG",
        metadata={"help": "The path to the output folder."}
    )

    save_intermediate: bool = field(
        default=True,
        metadata={"help": "Store or not the intermediate files, such as original dataset."}
    )

    model_path: str = field(
        default="./bart.large.cnn/models.pt",
        metadata={"help": "The path to store the models .pt checkpoint. The models is loaded before training"})

    cuda_devices: str = field(
        default="0,1",
        metadata={'help': "The index of GPUs used to train BART-large, seperated by , ."}
    )

    mode: str = field(
        default="train",
        metadata={"help": "Train the whole dataset or test on test set."}
    )

    checkpoint_dir: str = field(
        default="",
        metadata={"help": "The directory to save the checkpoints"}
    )

    # total_number_update: int = field(
    #     default=30000,
    #     metadata={"help": "Number of steps for training."})
    #
    # warmup_update: int = field(
    #     default=500,
    #     metadata={"help": "Number of warmup steps for training."})
    #
    # learning_rate: float = field(
    #     default=6e-5,
    #     metadata={"help": "Learning rate of training."})
    #
    # input_max_length: int = field(
    #     default=1024,
    #     metadata={"help": "Maximum number of tokens in the input."})
    #
    # update_freq: int = field(
    #     default=4,
    #     metadata={"help": "Update frequency of the models."})
    #
    # generate_batch_size: int = field(
    #     default=64,
    #     metadata={"help": "Number of generated sample per batch."}
    # )

    # arch: str = field(
    #     default="bart-large",
    #     metadata={"help": "Model architecture, such as bart-large, bart-base"})
