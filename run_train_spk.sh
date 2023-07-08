export MODEL=wav2vec2-base
export TOKENIZER=wav2vec2-base
export LR=5e-4
export ACC=8 # batch size * acc = 8
export WORKER_NUM=4
export MODEL_PATH=../ckpts/baseline
export WANDB_PROJECT=speaker_training
export ID=$1

export CUDA_VISIBLE_DEVICES="1, 2, 3"
export CUDA_LAUNCH_BLOCKING=1
export N_GPU=3

python -m torch.distributed.launch \
--nproc_per_node $N_GPU --use-env run_train_spk.py \
--output_dir=outputs/ckpts \
--cache_dir=cache/ \
--num_train_epochs=60 \
--per_device_train_batch_size="1" \
--per_device_eval_batch_size="1" \
--gradient_accumulation_steps=$ACC \
--dataset_name emotion \
--split_id $ID \
--evaluation_strategy=epoch \
--save_strategy=epoch \
--save_total_limit="2" \
--load_best_model_at_end=True \
--logging_dir=outputs/logs \
--report_to wandb \
--run_name train_spk_$ID \
--do_train \
--do_eval \
--learning_rate=$LR \
--model_name_or_path=$MODEL_PATH/id_$ID \
--tokenizer facebook/$TOKENIZER \
--preprocessing_num_workers=$WORKER_NUM \
--dataloader_num_workers $WORKER_NUM
# --freeze_feature_extractor \
# --gradient_checkpointing true \
# --fp16 \
