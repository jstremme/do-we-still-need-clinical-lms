#!/bin/sh
INPUT_MODEL="/Users/jstremme/models/RoBERTa-base-PM-M3-Voc-distill-align/RoBERTa-base-PM-M3-Voc-distill-align-hf"
OUT_DIR="../../model_outputs"
LR=0.00001
MAX_SEQ=512
STRIDE=128
BS=2
GAS=32

# use effective batch size of 64 per paper

echo 'Input: ' $INPUT_MODEL
echo 'Output: ' $OUT_DIR

for i in 1 2 # 3
do
    echo "Saving to: $OUT_DIR/seed_$i/" 
    python ../../src/finetuning/run_qa.py \
        --model_name_or_path $INPUT_MODEL \
        --train_file "../../data_preprocessed/radqa/train.csv" \
        --validation_file "../../data_preprocessed//radqa/dev.csv" \
        --test_file "../../data_preprocessed//radqa/test.csv" \
        --version_2_with_negative \
        --do_train \
        --do_eval \
        --do_predict \
        --per_device_train_batch_size $BS \
        --per_device_eval_batch_size $BS \
        --gradient_accumulation_steps $GAS \
        --learning_rate $LR \
        --num_train_epochs 10 \
        --save_total_limit 1 \
        --metric_for_best_model "eval_f1" \
        --report_to "wandb" \
        --evaluation_strategy "epoch" \
        --save_strategy "epoch" \
        --seed $i \
        --load_best_model_at_end \
        --overwrite_output_dir \
        --max_seq_length $MAX_SEQ \
        --doc_stride $STRIDE \
        --output_dir "$OUT_DIR/seed_$i/" \

done
