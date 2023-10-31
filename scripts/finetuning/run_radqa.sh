#!/bin/sh
INPUT_MODEL="/Users/jstremme/models/RoBERTa-base-PM-M3-Voc-distill-align/RoBERTa-base-PM-M3-Voc-distill-align-hf"
OUT_DIR="../../model_outputs"
LR=0.00001

echo 'Input: ' $INPUT_MODEL
echo 'Output: ' $OUT_DIR
echo 'LR: ' $LR

# use effective batch size of 64 per paper

for i in 1 2 3
do
    echo "Storing in $OUT_DIR/seed_$i/"
    python ../../src/finetuning/run_seq2seq_qa.py \
      --model_name_or_path $INPUT_MODEL \
      --train_file "../../data_preprocessed/radqa/train.csv" \
      --validation_file "../../data_preprocessed//radqa/dev.csv" \
      --test_file "../../data_preprocessed//radqa/test.csv" \
      --context_column context \
      --question_column question \
      --answer_column answers \
      --do_train \
      --do_eval \
      --do_predict \
      --save_total_limit 1 \
      --per_device_train_batch_size 2 \
      --gradient_accumulation_steps 32 \
      --per_device_eval_batch_size 1 \
      --num_train_epochs 15 \
      --version_2_with_negative \
      --max_seq_length 1024 \
      --load_best_model_at_end \
      --predict_with_generate \
      --metric_for_best_model "f1" \
      --report_to "wandb" \
      --evaluation_strategy "epoch" \
      --learning_rate $LR \
      --lr_scheduler_type "constant" \
      --optim "adafactor" \
      --save_strategy "epoch" \
      --overwrite_output_dir \
      --seed $i \
      --output_dir $OUT_DIR/seed_$i/
done
