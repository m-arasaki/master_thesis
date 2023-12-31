export ID=$1
export MODEL=facebook/wav2vec2-base
export TOKENIZER=facebook/wav2vec2-base
export LR=5e-4
export ACC=8 # batch size * acc = 8
export WORKER_NUM=4
export OUTPUT_DIR=outputs/speaker_$ID

python run_speaker.py \
--output_dir=$OUTPUT_DIR \
--logging_dir=$OUTPUT_DIR/logs \
--cache_dir=cache/ \
--num_train_epochs=3 \
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
