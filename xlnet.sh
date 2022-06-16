#ahotrod/xlnet_large_squad2_512

#export EXAMPLES=/private/wenlong/transformers/examples/question-answering
export SQUAD=/private/wenlong/squad/data
export CACHE_PATH=/private/wenlong/squad/output/xlnet
export MODEL_PATH=/private/wenlong/squad/output/electra_finetune_bs12_lr3e-6_ep3
export OUTPUT_PATH=/private/wenlong/squad/output/xlnet_eval
export CUDA_VISIBLE_DEVICES=2,3


python run_squad.py \
  --model_type xlnet \
  --model_name_or_path ahotrod/xlnet_large_squad2_512 \
  --cache_dir ${CACHE_PATH} \
  --do_eval \
  --train_file ${SQUAD}/merge_train.json \
  --predict_file ${SQUAD}/new_dev.json \
  --version_2_with_negative \
  --do_lower_case \
  --fp16 \
  --fp16_opt_level O1 \
  --num_train_epochs 3 \
  --warmup_steps 136 \
  --weight_decay 0.01 \
  --learning_rate 3e-6 \
  --max_grad_norm 0.5 \
  --adam_epsilon 1e-6 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --per_gpu_train_batch_size 12 \
  --gradient_accumulation_steps 16 \
  --per_gpu_eval_batch_size 128 \
  --threads 24 \
  --logging_steps 100 \
  --save_steps 500 \
  --eval_all_checkpoints \
  --evaluate_during_training \
  --overwrite_output_dir \
  --output_dir ${OUTPUT_PATH} \
  --seed 1996