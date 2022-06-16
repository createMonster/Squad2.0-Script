#ahotrod/albert_xxlargev1_squad2_512

export EXAMPLES=/private/wenlong/transformers/examples/question-answering
export SQUAD=/private/wenlong/squad/data
export CACHE_PATH=/private/wenlong/squad/output/albertxxlarge
export MODEL_PATH=/private/wenlong/squad/at/albert30
export OUTPUT_PATH=/private/wenlong/squad/class_output/albert_bs3_lr3e-6_ep3
export CUDA_VISIBLE_DEVICES=0,1


python run_at.py \
  --model_type albert \
  --model_name_or_path ${MODEL_PATH} \
  --do_train\
  --do_eval \
  --cache_dir ${CACHE_PATH} \
  --train_file ${SQUAD}/why.json \
  --predict_file ${SQUAD}/new_dev.json \
  --version_2_with_negative \
  --do_lower_case \
  --fp16 \
  --fp16_opt_level O1 \
  --num_train_epochs 3 \
  --warmup_steps 6 \
  --learning_rate 3e-6 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --per_gpu_train_batch_size 3 \
  --gradient_accumulation_steps 16 \
  --per_gpu_eval_batch_size 48 \
  --threads 24 \
  --logging_steps 150 \
  --save_steps 20 \
  --eval_all_checkpoints \
  --overwrite_output_dir \
  --output_dir ${OUTPUT_PATH}\