import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from torch import nn
from model import Wav2Vec2ForCTCnCLS

class Wav2Vec2AdversarialSpk(Wav2Vec2PreTrainedModel):

    mode = 'train_emotion'

    def __init__(self, config, emo_len=4, spk_len=8, beta=0.75):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.emo_head = nn.Sequential(
            nn.Linear(config.hidden_size, emo_len)
        )
        self.spk_head = nn.Sequential(
            nn.Linear(config.hidden_size, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, spk_len)
        )
        self.init_weights()
        self.beta = beta
        

    def freeze_feature_extractor(self):
        for p in self.wav2vec2.parameters():
            p.requires_grad = False
    
    def freeze_emo_head(self): 
        for param in self.emo_head.parameters():
            param.requires_grad = False
        
        print('<debug:> successfully freezed cls_head')
    
    def freeze_spk_head(self): 
        for param in self.spk_head.parameters():
            param.requires_grad = False
        
        print('<debug:> successfully freezed spk_head')
    
    def _cls_loss(self, logits, cls_labels): # sum hidden_states over dim 1 (the sequence length), then feed into self.cls
        loss = None
        if cls_labels is not None:
            loss = F.cross_entropy(logits, cls_labels.to(logits.device))
        return loss

    def _entropy(self, probs):
        probs = torch.clamp(probs, min=torch.finfo(probs.dtype).eps)
        return torch.mean(torch.sum(-probs * torch.log(probs), dim=1)) 

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None, 
        ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 特徴量抽出器
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        # logits. lossの計算
        logits, loss = None, None
        if self.mode == 'train_speaker': 
            # 話者認識器訓練モード
            logits = self.spk_head(torch.mean(hidden_states, dim=1))
            loss = self._cls_loss(logits, labels)
        elif self.mode == 'train_emotion':
            # 感情認識器訓練モード

            # 各logitsを計算
            logits_emo = self.emo_head(torch.mean(hidden_states, dim=1)) 
            logits_spk = self.spk_head(torch.mean(hidden_states, dim=1))
            probs_spk = F.softmax(logits_spk, dim=-1) # 話者認識については，probsに意味があるためlogitsではなくprobを扱う

            # 各lossを計算
            loss_emo = self._cls_loss(logits_emo, labels)
            loss_spk_H = self._entropy(probs_spk)

            # 全体のlogits, lossを計算
            logits = (logits_emo, probs_spk)
            loss = self.beta * loss_emo - (1 - self.beta) * loss_spk_H
        elif self.mode == 'eval': 
            # 評価モード
            logits = self.emo_head(torch.mean(hidden_states, dim=1))
            loss = self._cls_loss(logits, labels)
        else:
            raise Exception('Invalid mode: ' + self.mode)
            
        return ModelOutput(
            loss=loss, logits=logits, hidden_states=hidden_states
        )