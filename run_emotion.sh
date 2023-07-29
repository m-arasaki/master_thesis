export ID=$1
export MODEL=./outputs/speaker_$ID
export TOKENIZER=facebook/wav2vec2-base
export LR=5e-5
export ACC=8 # batch size * acc = 8
export WORKER_NUM=4
export OUTPUT_DIR=outputs/emotion_$ID


python run_emotion.py \
--model_name_or_path $MODEL \
--tokenizer $TOKENIZER \
--split_id $ID \
--cache_dir=cache/ \
--beta 0.99 \
--max_duration_in_seconds=17 \
--output_dir=$OUTPUT_DIR \
--logging_dir=$OUTPUT_DIR/logs \
--num_train_epochs=3 \
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