import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from torch import nn
from model import Wav2Vec2ForCTCnCLS

class Wav2Vec2AdversarialSpk(Wav2Vec2ForCTCnCLS):

    def __init__(self, config, emo_len=4, spk_len=8, beta=0.5):
        super().__init__(config)
        # emotion classifier is defined as cls_head in superclass
        self.spk_head = nn.Sequential(
            nn.Linear(config.hidden_size, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, spk_len)
        )
        self.beta = beta

    def freeze_feature_extractor(self):
        for p in self.wav2vec2.parameters():
            p.requires_grad = False
    
    def freeze_cls_head(self): # cls_headの重みを凍結
        for param in self.cls_head.parameters():
            param.requires_grad = False
        
        print('<debug:> successfully freezed cls_head')
    
    def freeze_spk_head(self): # spk_headの重みを凍結
        for param in self.spk_head.parameters():
            param.requires_grad = False
        
        print('<debug:> successfully freezed spk_head')

    def _max_entropy_loss(self, logits):
        probs = F.softmax(logits)
        return - torch.mean(torch.sum(-probs * torch.log(probs), dim=1)) # エントロピーの符号反転を出力

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None, # tuple: (emo_labels, spk_labels) shape\
        if_train_spk = True, # 
        if_train_emo = False, 
        if_eval = False
        ):
        # if_train_spk, if_train_emo, if_eval: 2 out of 3 should not be True

        if if_train_emo + if_train_spk + if_eval > 1:
            raise Exception(f'2 out of 3 should not be True: if_train_spk:{if_train_spk}, if_train_emo:{if_train_emo}, if_eval:{if_eval}')

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0] # this is the last layer's hidden states
        hidden_states = self.dropout(hidden_states)

        logits_emo = self.cls_head(torch.mean(hidden_states, dim=1)) if if_train_emo or if_eval else None # クラス分類タスクの際は平均を取る
        logits_spk = self.spk_head(torch.mean(hidden_states, dim=1)) if if_train_emo or if_train_spk else None

        loss= None
        if labels is not None:
            if if_train_spk: # assume labels are speaker labels
                loss = self._cls_loss(logits_spk, labels)
            if if_train_emo: # assume labels are emotion labels
                loss_emo = self._cls_loss(logits_emo, labels)
                loss_spk_H = self._max_entropy_loss(logits_spk)

                loss = self.beta * loss_emo - (1 - self.beta) * loss_spk_H
            
            if if_eval: # assume labels are emotion labels
                loss = self._cls_loss(logits_emo, labels)
        
        logits = None
        if if_train_emo:
            logits = (logits_emo, logits_spk)
        elif if_train_spk:
            logits = logits_spk
        else:
            logits = logits_emo
            
        return ModelOutput(
            loss=loss, logits=logits, hidden_states=hidden_states
        )