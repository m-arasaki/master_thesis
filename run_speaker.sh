export ID=$1
export MODEL=../ckpts/baseline/screened17/id_$ID
export TOKENIZER=facebook/wav2vec2-base
export LR=5e-4
export ACC=8 # batch size * acc = 8
export WORKER_NUM=4
export WANDB_PROJECT=speaker_from_baseline

export CUDA_VISIBLE_DEVICES="0, 1, 2, 3"
export CUDA_LAUNCH_BLOCKING=1
export N_GPU=4

python -m torch.distributed.launch \
--nproc_per_node $N_GPU --use-env run_speaker.py \
--output_dir=$WANDB_PROJECT \
--logging_dir=$WANDB_PROJECT/logs \
--cache_dir=cache/ \
--num_train_epochs=50 \
--per_device_train_batch_size="1" \
--per_device_eval_batch_size="1" \
--max_duration_in_seconds=17 \
--gradient_accumulation_steps=$ACC \
--dataset_name emotion \
--split_id $ID \
--evaluation_strategy=epoch \
--save_strategy=epoch \
--save_total_limit="1" \
--load_best_model_at_end=True \
--metric_for_best_model=eval_acc \
--report_to wandb \
--run_name id_$ID \
--do_train \
--do_eval \
--learning_rate=$LR \
--model_name_or_path=$MODEL \
--tokenizer $TOKENIZER \
--preprocessing_num_workers=$WORKER_NUM \
--dataloader_num_workers $WORKER_NUM
# --freeze_feature_extractor \
# --gradient_checkpointing true \
# --fp16 \
