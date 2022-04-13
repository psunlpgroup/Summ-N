#!/usr/bin/env bash


INPUT=$1
OUTPUT=$2

mkdir "$OUTPUT"
for SPLIT in train val
do
for LANG in source target
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "$INPUT/$SPLIT.$LANG" \
    --outputs "$OUTPUT/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done

fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "${OUTPUT}/train.bpe" \
  --validpref "${OUTPUT}/val.bpe" \
  --destdir "${OUTPUT}/bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;

conda activate fairseq

TOTAL_NUM_UPDATES=30000
WARMUP_UPDATES=500
LR=3e-05
MAX_TOKENS=2048
UPDATE_FREQ=4

BART_PATH=./bart.large.cnn/model.pt

CUDA_VISIBLE_DEVICES=$3 fairseq-train "${OUTPUT}/bin/" \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --total-num-update $TOTAL_NUM_UPDATES\
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR \
    --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --no-epoch-checkpoints \
    --arch bart_large \
    --save-dir "${OUTPUT}/checkpoints"\
    --patience 2;
    # --max-source-positions $MAX_PER_SAMPLE
    # --arch bart_base;

