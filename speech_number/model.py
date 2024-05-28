from dataclasses import dataclass
import json
import os
from typing import Optional

import numpy as np
import tqdm
from transformers.modeling_outputs import ModelOutput
from transformers.models.whisper.modeling_whisper import WhisperEncoder, WhisperPreTrainedModel
import torch
import torchaudio
from torch import nn

from speech_number.base_model import BaseModel
from sklearn.metrics import accuracy_score, f1_score, classification_report

@dataclass
class WhisperEncoderOutput(ModelOutput):
    encoder_hidden_states: torch.FloatTensor = None
    cls_hidden_states: torch.FloatTensor = None
    cls_logits: torch.FloatTensor = None
    loss: torch.FloatTensor = None


class WhisperEncoderCustomize(WhisperPreTrainedModel, BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = WhisperEncoder(config)
        self.matrix_transpose = torch.nn.Linear(1500, 1)
        self.norm = nn.BatchNorm1d(384)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(384, 22)

    def forward(self, input_features, labels):
        # Whisper encoder
        encoder_outputs = self.encoder(input_features) 
        encoder_hidden_states = encoder_outputs.last_hidden_state # (N, 1500, 384)
        # Dropout
        x = self.dropout(encoder_hidden_states)

        x = x.transpose(1, 2) # (N, 384, 1500)
        x = self.matrix_transpose(x) # (N, 384, 1)
        x = self.norm(x)
        cls_hidden_states  = x.transpose(1, 2) # (N, 1, 384)


        x = self.tanh(cls_hidden_states)
        x = self.dropout(x)
        x = self.out_proj(x) # (N, 1, 22)
        cls_logits = x.squeeze(1)
        
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(cls_logits, labels)
        else:
            loss = 0

        return WhisperEncoderOutput(
            encoder_hidden_states=encoder_hidden_states,
            cls_hidden_states=cls_hidden_states,
            cls_logits=cls_logits,
            loss=loss
        )

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def pre_epoch(self, trainer, epoch):
        if epoch < 8:
            self.freeze_encoder()
        else:
            self.unfreeze_encoder()

    def train_epoch(self, trainer, epoch, outfile):
        self.train()

        # init
        log_info = {
            "loss": []
        }

        # loop
        for batch_data in tqdm.tqdm(trainer.dataloader_train, ncols=100, desc=f"Train epoch {epoch}"):
            input_features = batch_data["input_features"].to(self.device)
            labels = batch_data["labels"].to(self.device)
            B = len(labels)

            # forward
            output = self.forward(input_features, labels)

            # backward
            loss = output.loss
            loss.backward()
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()

            # log step loss
            log_info["loss"].append(loss.item())

        # log epoch loss
        log_info["loss"] = round(np.mean(log_info["loss"]), 3) if log_info["loss"] else 0

        return log_info

    def test_epoch(self, trainer, dataloader, epoch, data_name, outfile):
        self.eval()
        # init
        log_info = {
            "loss": []
        }

        list_labels = []
        list_predicts = []
        # loop
        for batch_data in tqdm.tqdm(dataloader, ncols=100, desc=f"Test {data_name} epoch {epoch}"):
            input_features = batch_data["input_features"].to(self.device)
            labels = batch_data["labels"].to(self.device)

            # forward
            with torch.no_grad():
                output = self.forward(input_features, labels)

            # backward
            loss = output.loss
            labels = labels.cpu().tolist()
            predicts = torch.argmax(output.cls_logits.detach().cpu(), dim=-1).tolist()

            # log step loss
            list_labels += labels
            list_predicts += predicts
            log_info["loss"].append(loss.item())

        acc, f1 = self.metrics_compute(list_labels, list_predicts, epoch, is_train=False, out_file=outfile)

        log_info["loss"] = round(np.mean(log_info["loss"]), 3) if log_info["loss"] else 0
        log_info["accuracy"] = acc
        log_info["f1_score"] = f1

        log_info_json = json.dumps(log_info, indent=4)
        print(log_info)
        print(log_info_json, file=outfile)
        return log_info
    
    def metrics_compute(self, true_labels, predicted_labels, epoch, is_train = True, out_file=None):
        if is_train:
            print(f"---Train epoch {epoch}---", file=out_file)
        else:
            print(f"---Test epoch {epoch}---", file=out_file)
        acc = round(accuracy_score(y_true=true_labels, y_pred=predicted_labels), 3)
        f1 = round(f1_score(y_true=true_labels, y_pred=predicted_labels, average="micro"), 3)
        if epoch % 5 == 0:
            classify_report = classification_report(true_labels, predicted_labels, labels=list(range(22)), zero_division=0)
            print(classify_report, file=out_file)
        return acc, f1

    def get_param_size(self):
        '''
        In ra số lượng tham số  của mô hình và kích thước, trọng lượng của mô hình (MB)
        '''
        total_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_size = sum(p.numel() * p.element_size() for p in self.parameters())

        total_size_mb = total_size / (1024 * 1024)
        print(f"Số lượng tham số : {round(total_param/1000000, 3)}M . Kích thước mô hình : {round(total_size_mb, 3)} MB")

    def predict(self, feature_extractor, model, index2label, input_audio, sample_rate=None):
        if isinstance(input_audio, str):
            # assert not os.path.exists(input_audio), f"Đường dẫn {input_audio} không tồn tại"
            waveform, sample_rate = torchaudio.load(input_audio)
        else:
            waveform = torch.FloatTensor(input_audio).view(1, -1)
            
        new_sample_rate = 16000
        waveform = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform)
        waveform = waveform.squeeze()
        input_features = feature_extractor(waveform, sampling_rate=new_sample_rate, return_tensors="pt").input_features
        input_features = input_features.to(self.device)
        with torch.no_grad():
            output = model(input_features, labels=None)
            predicts = torch.argmax(output.cls_logits.detach().cpu(), dim=-1).tolist()

        labels_predict = index2label.get(str(predicts[0]), -1)
        return labels_predict

if __name__ == "__main__":
    model_name = "vinai/PhoWhisper-tiny"
    whisper_encoder = WhisperEncoderCustomize.from_pretrained(model_name)
    whisper_encoder.get_param_size()
    print(whisper_encoder)

