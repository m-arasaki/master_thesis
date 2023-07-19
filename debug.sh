export MODEL=../ckpts/train_spk
export TOKENIZER=wav2vec2-base
export LR=5e-5
export ACC=6 # batch size * acc = 8
export WORKER_NUM=4
export ID=$1
export WANDB_PROJECT=$2

export CUDA_VISIBLE_DEVICES="0, 1, 2, 3"
export CUDA_LAUNCH_BLOCKING=1
export N_GPU=4

python -m torch.distributed.launch \
--nproc_per_node $N_GPU --use_env run_emotion.py \
--model_name_or_path $MODEL/id_$ID \
--split_id $ID \
--cache_dir=cache/ \
--beta 0.75 \
--output_dir=outputs/train_emotion \
--logging_dir=outputs/train_emotion \
--tokenizer facebook/$TOKENIZER \
--num_train_epochs=50 \
--per_device_train_batch_size="1" \
--per_device_eval_batch_size="1" \
--gradient_accumulation_steps=$ACC \
--dataset_name emotion \
--evaluation_strategy=steps \
--save_strategy=steps \
--eval_steps=100 \
--run_name train_emotion_$ID \
--load_best_model_at_end=True \
--metric_for_best_model=eval_acc \
--save_total_limit="1" \
--do_train \
--do_eval \
--learning_rate=$LR \
--preprocessing_num_workers=$WORKER_NUM \
--dataloader_num_workers $WORKER_NUM
--mode train_emotion \
--report_to wandb \
# --freeze_feature_extractor \
# --gradient_checkpointing true \
# --fp16 \
