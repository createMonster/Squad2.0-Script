#phiyodr/roberta-large-finetuned-squad2
#google/electra-large-discriminator

#export EXAMPLES=/private/wenlong/transformers/examples/question-answering
export SQUAD=/private/wenlong/squad/data
export CACHE_PATH=/private/wenlong/squad/output/roberta
export MODEL_PATH=/private/wenlong/squad/output/roberta_finetune_bs8_lr6e-6_ep3
export OUTPUT_PATH=/private/wenlong/squad/last_shine/roberta_finetune_bs8_lr6e-6_ep3_dev
export CUDA_VISIBLE_DEVICES=0,1


python run_squad.py \
  --model_type roberta \
  --model_name_or_path phiyodr/roberta-large-finetuned-squad2 \
  --cache_dir ${CACHE_PATH} \
  --do_train \
  --do_eval \
  --train_file ${SQUAD}/merge_train.json \
  --predict_file ${SQUAD}/new_dev.json \
  --version_2_with_negative \
  --do_lower_case \
  --fp16 \
  --fp16_opt_level O1 \
  --num_train_epochs 3 \
  --warmup_steps 196 \
  --weight_decay 0.01 \
  --learning_rate 6e-6 \
  --max_grad_norm 0.5 \
  --adam_epsilon 1e-6 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --per_gpu_train_batch_size 8 \
  --gradient_accumulation_steps 16 \
  --per_gpu_eval_batch_size 128 \
  --threads 24 \
  --logging_steps 100 \
  --save_steps 800 \
  --eval_all_checkpoints \
  --overwrite_output_dir \
  --output_dir ${OUTPUT_PATH}