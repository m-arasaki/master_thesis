#!/usr/bin/env python3
import logging
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Union

import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

import librosa
from lang_trans import arabic

import soundfile as sf
from model import Wav2Vec2ForCTCnCLS
from proposed_model import Wav2Vec2AdversarialSpk
from transformers.trainer_utils import get_last_checkpoint
import os
import datetime
import wandb
import pickle

from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    is_apex_available,
    trainer_utils,
    EarlyStoppingCallback
)


if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_feature_extractor: Optional[bool] = field(
        default=True, metadata={"help": "Whether to freeze the feature extractor layers of the model."}
    )
    verbose_logging: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to log verbose messages or not."},
    )
    tokenizer: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained tokenizer"}
    )


def configure_logger(model_args: ModelArguments, training_args: TrainingArguments):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging_level = logging.WARNING
    if model_args.verbose_logging:
        logging_level = logging.DEBUG
    elif trainer_utils.is_main_process(training_args.local_rank):
        logging_level = logging.INFO
    logger.setLevel(logging_level)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: str = field(
        default='emotion', metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_split_name: Optional[str] = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    validation_split_name: Optional[str] = field(
        default="validation",
        metadata={
            "help": "The name of the validation data set split to use (via the datasets library). Defaults to 'validation'"
        },
    )
    target_text_column: Optional[str] = field(
        default="text",
        metadata={"help": "Column in the dataset that contains label (target text). Defaults to 'text'"},
    )
    speech_file_column: Optional[str] = field(
        default="file",
        metadata={"help": "Column in the dataset that contains speech file path. Defaults to 'file'"},
    )
    target_feature_extractor_sampling_rate: Optional[bool] = field(
        default=False,
        metadata={"help": "Resample loaded audio to target feature extractor's sampling rate or not."},
    )
    max_duration_in_seconds: Optional[float] = field(
        default=None,
        metadata={"help": "Filters out examples longer than specified. Defaults to no filtering."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    # select which split as test
    split_id: int = field(
        default='1', metadata={"help": "iemocap_ + splitid (e.g. 1, 2, etc) + train/val/test.csv"}
    )

    output_file: Optional[str] = field(
        default=None,
        metadata={"help": "Output file."},
    )

@dataclass
class MyDataCollator:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    audio_only = False

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        cls_labels = [feature["labels"][-1] for feature in features]
        
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(cls_labels) # labels = (ctc_labels, cls_labels)
        return batch


class MyTrainer(Trainer):
    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                kwargs = dict(device=self.args.device)
                if self.deepspeed and inputs[k].dtype != torch.int64:
                    kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))
                inputs[k] = v.to(**kwargs)

            if k == 'labels': # labels are list of tensor, not tensor, special handle here
                for i in range(len(inputs[k])):
                    kwargs = dict(device=self.args.device)
                    if self.deepspeed and inputs[k][i].dtype != torch.int64:
                        kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))
                    inputs[k][i] = inputs[k][i].to(**kwargs)

        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

def create_processor(model_args: ModelArguments) -> Wav2Vec2Processor:
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir
    )
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
        model_args.tokenizer,
        cache_dir=model_args.cache_dir,
        word_delimator_token='|',
        do_lower_case=False
    )
    return Wav2Vec2Processor(feature_extractor, tokenizer)

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dt_now = datetime.datetime.now()
    id_str = dt_now.strftime('%Y%m%d_%H%M') + '_id_' + str(data_args.split_id)
    training_args.output_dir = os.path.join(training_args.output_dir, 'output' + id_str)
    training_args.logging_dir = os.path.join(training_args.logging_dir, 'logs' + id_str)
    configure_logger(model_args, training_args)

    processor = create_processor(model_args)
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    if data_args.dataset_name == 'emotion':
        dataset = datasets.load_dataset('csv', data_files='../dataset/iemocap/iemocap_' + str(data_args.split_id) + '.train.csv', cache_dir=model_args.cache_dir)['train'].train_test_split(test_size=0.1)
        train_dataset = dataset['train']
        val_dataset = dataset['test']

    spk_set = sorted(set(train_dataset['speaker']))
    spk_ids = [i for i in range(len(spk_set))]
    speaker_dict = dict(zip(spk_set, spk_ids))
    spk_len = len(speaker_dict)

    print('<debug:> showing the speaker_dict')
    print(speaker_dict)

    model = Wav2Vec2AdversarialSpk.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        spk_len=spk_len,
    )
    model.config.gradient_checkpointing = True
    model.freeze_emo_head()
    model.freeze_feature_extractor()
    model.mode = 'train_speaker'

    target_sr = processor.feature_extractor.sampling_rate if data_args.target_feature_extractor_sampling_rate else None
    def prepare_example(example, audio_only=False):  # TODO(elgeish) make use of multiprocessing?
        example["speech"], example["sampling_rate"] = librosa.load(example[data_args.speech_file_column], sr=target_sr)
        if data_args.max_duration_in_seconds is not None:
            example["duration_in_seconds"] = len(example["speech"]) / example["sampling_rate"]
        return example

    if training_args.do_train:
        train_dataset = train_dataset.map(prepare_example, remove_columns=[data_args.speech_file_column])
    if training_args.do_predict:
        val_dataset = val_dataset.map(prepare_example, fn_kwargs={'audio_only':True})
    elif training_args.do_eval:
        val_dataset = val_dataset.map(prepare_example, remove_columns=[data_args.speech_file_column])

    if data_args.max_duration_in_seconds is not None:
        def filter_by_max_duration(example):
            return example["duration_in_seconds"] <= data_args.max_duration_in_seconds
        if training_args.do_train:
            old_train_size = len(train_dataset)
            train_dataset = train_dataset.filter(filter_by_max_duration, remove_columns=["duration_in_seconds"])
            if len(train_dataset) > old_train_size:
                logger.warning(
                    f"Filtered out {len(train_dataset) - old_train_size} train example(s) longer than {data_args.max_duration_in_seconds} second(s)."
                )
        if training_args.do_predict or training_args.do_eval:
            old_val_size = len(val_dataset)
            val_dataset = val_dataset.filter(filter_by_max_duration, remove_columns=["duration_in_seconds"])
            if len(val_dataset) > old_val_size:
                logger.warning(
                    f"Filtered out {len(val_dataset) - old_val_size} validation example(s) longer than {data_args.max_duration_in_seconds} second(s)."
                )
    # logger.info(f"Split sizes: {len(train_dataset)} train and {len(val_dataset)} validation.")

    def prepare_dataset(batch, audio_only=False):
        # check that all files have the correct sampling rate
        assert (
            len(set(batch["sampling_rate"])) == 1
        ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."
        batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
        if audio_only is False:
            cls_labels = list(map(lambda s: speaker_dict[s], batch["speaker"]))
            batch["labels"] = [[] * i for i in range(len(cls_labels))]
            for i in range(len(cls_labels)):
                batch["labels"][i].append(cls_labels[i]) # batch["labels"] element has to be a single list
        return batch

    cols_to_remove = ['Unnamed: 0.1', 'Unnamed: 0', 'emotion', 'speaker', 'text', 'speech', 'sampling_rate']
    if training_args.do_train:
        train_dataset = train_dataset.map(
            prepare_dataset,
            remove_columns=cols_to_remove,
            batch_size=training_args.per_device_train_batch_size,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
        )
    if training_args.do_predict:
        val_dataset = val_dataset.map(
            prepare_dataset,
            remove_columns=cols_to_remove,
            fn_kwargs={'audio_only':True},
            batch_size=training_args.per_device_train_batch_size,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
        )
    elif training_args.do_eval:
        val_dataset = val_dataset.map(
            prepare_dataset,
            remove_columns=cols_to_remove,
            batch_size=training_args.per_device_train_batch_size,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
        )
    

    data_collator = MyDataCollator(processor=processor, padding=True)

    def compute_metrics(pred):
        cls_pred_logits = pred.predictions[0]
        cls_pred_ids = np.argmax(cls_pred_logits, axis=-1)
        total = len(pred.label_ids)
        correct = (cls_pred_ids == pred.label_ids).sum().item() # label = (ctc_label, cls_label)
        return {"acc": correct/total, "correct": correct, "total": total}

    trainer = MyTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
        checkpoint = model_args.model_name_or_path
    else:
        checkpoint = None

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model() 

    if training_args.do_predict:
        logger.info('******* Predict ********')
        data_collator.audio_only=True
        predictions, labels, metrics = trainer.predict(val_dataset, metric_key_prefix="predict")
        logits_ctc, logits_cls = predictions
        pred_ids = np.argmax(logits_cls, axis=-1)
        pred_probs = F.softmax(torch.from_numpy(logits_cls).float(), dim=-1)
        print(val_dataset)
        with open(data_args.output_file, 'w') as f:
            for i in range(len(pred_ids)):
                f.write(val_dataset[i]['file'].split("/")[-1] + " " + str(len(val_dataset[i]['input_values'])/16000) + " ")
                pred = pred_ids[i]
                f.write(str(pred)+' ')
                for j in range(4):
                    f.write(' ' + str(pred_probs[i][j].item()))
                f.write('\n')
        f.close()

    elif training_args.do_eval:
        predictions, labels, metrics = trainer.predict(val_dataset, metric_key_prefix="eval")
        print(metrics)

if __name__ == "__main__":
    main()
