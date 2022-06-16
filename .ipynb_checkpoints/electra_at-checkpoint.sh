#ahotrod/electra_large_discriminator_squad2_512
#google/electra-large-discriminator
#gradient_accumulation_steps 16


#export EXAMPLES=/private/wenlong/transformers/examples/question-answering
export SQUAD=/private/wenlong/squad/data
export CACHE_PATH=/private/wenlong/squad/output/electra
export MODEL_PATH=/private/wenlong/squad/class_output/electra30
export OUTPUT_PATH=/private/wenlong/squad/at/electra30_lr3e-6_bs12_ep3_ga16_seed1996_epsi025
export CUDA_VISIBLE_DEVICES=0,1,2,3


python run_at.py \
  --model_type electra \
  --model_name_or_path ahotrod/electra_large_discriminator_squad2_512 \
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
  --warmup_steps 86 \
  --weight_decay 0.01 \
  --learning_rate 3e-6 \
  --max_grad_norm 0.5 \
  --adam_epsilon 1e-6 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --per_gpu_train_batch_size 12 \
  --gradient_accumulation_steps 16 \
  --per_gpu_eval_batch_size 96 \
  --threads 24 \
  --logging_steps 50 \
  --save_steps 200 \
  --eval_all_checkpoints \
  --evaluate_during_training \
  --overwrite_output_dir \
  --output_dir ${OUTPUT_PATH} \
  --seed 1996