export ID=$1
export MODEL=facebook/wav2vec2-base
export TOKENIZER=facebook/wav2vec2-base
export LR=5e-5
export ACC=6 # batch size * acc = 6
export WORKER_NUM=4
export WANDB_PROJECT=emotion_beta0.99again

export CUDA_VISIBLE_DEVICES="0, 1, 2, 3"
export CUDA_LAUNCH_BLOCKING=1
export N_GPU=4

python -m torch.distributed.launch \
--nproc_per_node $N_GPU --use_env run_emotion.py \
--model_name_or_path $MODEL \
--tokenizer $TOKENIZER \
--split_id $ID \
--cache_dir=cache/ \
--beta 0.99 \
--max_duration_in_seconds=17 \
--output_dir=projects/$WANDB_PROJECT \
--logging_dir=projects/$WANDB_PROJECT/logs \
--num_train_epochs=50 \
--per_device_train_batch_size="1" \
--per_device_eval_batch_size="1" \
--gradient_accumulation_steps=$ACC \
--dataset_name emotion \
--evaluation_strategy=epoch \
--save_strategy=epoch \
--run_name id_$ID \
--load_best_model_at_end=True \
--metric_for_best_model=eval_acc \
--save_total_limit="1" \
--do_train \
--do_eval \
--learning_rate=$LR \
--preprocessing_num_workers=$WORKER_NUM \
--dataloader_num_workers $WORKER_NUM \
--mode train_emotion \
--report_to wandb \
