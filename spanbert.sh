#mrm8488/spanbert-large-finetuned-squadv2

export SQUAD=/private/wenlong/squad/data
export CACHE_PATH=/private/wenlong/squad/output/spanbert
export MODEL_PATH=/private/wenlong/squad/output/albertxxlarge
export OUTPUT_PATH=/private/wenlong/squad/output/spanbert_eval
export CUDA_VISIBLE_DEVICES=0,1


python run_squad.py \
  --model_type bert \
  --model_name_or_path ${CACHE_PATH} \
  --do_train \
  --do_eval \
  --cache_dir ${CACHE_PATH} \
  --train_file ${SQUAD}/merge_train.json \
  --predict_file ${SQUAD}/new_dev.json \
  --version_2_with_negative \
  --fp16 \
  --fp16_opt_level O1 \
  --num_train_epochs 3 \
  --warmup_steps 312 \
  --learning_rate 3e-6 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --per_gpu_train_batch_size 6 \
  --gradient_accumulation_steps 16 \
  --per_gpu_eval_batch_size 48 \
  --threads 24 \
  --logging_steps 150 \
  --save_steps 1000 \
  --eval_all_checkpoints \
  --overwrite_output_dir \
  --output_dir ${OUTPUT_PATH}\