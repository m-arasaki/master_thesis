export MODEL=../ckpts/train_spk
export TOKENIZER=wav2vec2-base
export LR=5e-5
export ACC=6 # batch size * acc = 8
export WORKER_NUM=4
export ID=$1
export WANDB_PROJECT=emotion_training

export CUDA_VISIBLE_DEVICES="0, 1, 2, 3"
export CUDA_LAUNCH_BLOCKING=1
export N_GPU=4

python -m torch.distributed.launch \
--nproc_per_node $N_GPU --use_env run_emotion.py \
--model_name_or_path $MODEL/id_$ID \
--output_dir=output/tmp \
--cache_dir=cache/ \
--tokenizer facebook/$TOKENIZER \
--num_train_epochs=80 \
--per_device_train_batch_size="1" \
--per_device_eval_batch_size="1" \
--gradient_accumulation_steps=$ACC \
--dataset_name emotion \
--evaluation_strategy=epoch \
--save_strategy=epoch \
--run_name train_emotion_$ID \
--load_best_model_at_end=True \
--metric_for_best_model=eval_loss \
--save_total_limit="2" \
--do_train \
--do_eval \
--learning_rate=$LR \
--preprocessing_num_workers=$WORKER_NUM \
--dataloader_num_workers $WORKER_NUM
--mode train_emotion \
# --freeze_feature_extractor \
# --gradient_checkpointing true \
# --fp16 \
