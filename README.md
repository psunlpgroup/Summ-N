# SummN
Source code for ACL 2022 paper [SUMM^N: A Multi-Stage Summarization Framework for Long InputDialogues and Documents](https://arxiv.org/pdf/2110.10150.pdf)

## Dependency

- Install Fairseq according to their official instructions https://github.com/pytorch/fairseq
- `pip install -r requirements.txt` to install the rest of the packages
- We use python==3.7, pytorch==1.8.1, and fairseq==1.10.0

## Folder Structure

- configure: the running configures for each dataset, such as number of stages, beam width etc.
- dataset_loader: the python scripts to convert original dataset to the uniform format.
- models: SummN model
  - data_segment: including source and target segmentation code;
  - gen_summary: inference on the source text and generate coarse summaries;
  - train_summarizor.sh: we use fairseq-train command to train the model.
- scripts: all scripts to run experiments on different datasets.
- utils: utilities such as config parser & dataset reader etc.
- run.py the entrance of the code.

## Training and Evaluation

### Download the Datasets and Models
- Download link for AMI & ICSI can be found at https://github.com/microsoft/HMNet
- Download QMSum dataset from https://github.com/Yale-LILY/QMSum
- Download SummScreen (both MG and TMS) from https://github.com/mingdachen/SummScreen
- Download GovReport dataset from https://github.com/luyang-huang96/LongDocSum/tree/main/Model
- Run the following commands to download Fairseq BART-large models
```shell
# bart cnn
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.cnn.tar.gz
tar -xzvf bart.large.cnn.tar.gz

wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'
```
- Setup the ROUGE155 following https://github.com/chatc/AnyROUGE

### Training the Model
- After we setup the datasets, setup the paths of scripts at `scripts/{dataset name}.sh`
- Train the model by the command: `bash scripts/{dataset name}.sh`

### Evaluation
- First download the checkpoint from [Google Drive](https://drive.google.com/drive/folders/1pt_hsyKsBwL5l-iIwsz1ijvxV8v6fc1O?usp=sharing)
- Then, setup the paths of scripts at `scripts/{dataset name}.sh`
- Finally, specify the mode and checkpoint_dir in the running scripts. For instance,
```shell
python run.py --cfg ICSI.cfg \
 --dataset-path /data/yfz5488/fair/ICSI/ICSI_proprec \
 --output-path ./output/${RUN_NAME} \
 --save-intermediate \
 --cuda-devices 3 \
 --model-path $BART_PATH \
 --mode test \
 --checkpoint-dir path/to/checkpoints
```
And run this script to do the evaluation on test set only.

## Add a New Task
It is easy to add new task/dataset into Summ-N.
- First, add the configuration file in `configure` directory, one can write the cfg file following other files, e.g. `configure/ICSI.cfg` is a 3 stage config
- Then write the dataset loader and add it to `dataset_loader` directory. `dataset_loader/ICSI.py` can be a good example
- Finally, add the running parameters into `scripts`, following e.g. `scripts/run_ICSI.sh`
- Run the training or evaluation by `bash scripts/{Your Dataset}.sh`

## Citation
```bibtex
@inproceedings{zhang2021summn,
  title={Summ\^{} N: A Multi-Stage Summarization Framework for Long Input Dialogues and Documents},
  author={Zhang, Yusen and Ni, Ansong and Mao, Ziming and Wu, Chen Henry and Zhu, Chenguang and Deb, Budhaditya and Awadallah, Ahmed H and Radev, Dragomir and Zhang, Rui},
  booktitle={ACL 2022},
  year={2022}
}
``` 
