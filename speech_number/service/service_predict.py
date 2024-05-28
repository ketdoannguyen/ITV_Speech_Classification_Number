import json
from typing import Dict

from fastapi import FastAPI
import uvicorn

import torch
from speech_number.service.service_class import InferenceRequest, InferenceResponse
from speech_number.model import WhisperEncoderCustomize
from transformers import WhisperFeatureExtractor


class ClsNumber():
    def __init__(self, checkpoint_dir):
        self.model = WhisperEncoderCustomize.from_pretrained(checkpoint_dir)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("vinai/PhoWhisper-tiny")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.index2label = self.load_vocab()
        
    def load_vocab(self):
        with open('vocab_dataset.json', 'r') as file:
            data = json.load(file)
        return data["index2label"]
    
    def run(self, input_audio, sample_rate=None):
        return self.model.predict(self.feature_extractor, self.model, self.index2label, input_audio, sample_rate)

def start_aap_service(checkpoint_dir):
    fast_infer = FastAPI()
    cls_number = ClsNumber(checkpoint_dir)

    @fast_infer.post("/cls_number/infer")
    async def infer(data: InferenceRequest) -> InferenceResponse:
        # read data
        print("-"*100)
        
        input_audio = data.input_audio
        sample_rate = data.sample_rate # if sample_rate in data.keys() else None

        assert input_audio is not None, "Không có input của audio để inference"
        # run model
        model_outputs = cls_number.run(input_audio, sample_rate)

        response = {
            "label": model_outputs    
        }

        return response

    uvicorn.run(fast_infer)
    
def infer(checkpoint_dir, data: InferenceRequest) -> InferenceResponse:
    cls_number = ClsNumber(checkpoint_dir)
    # read data
    input_audio = data["input_audio"]
    sample_rate = data["sample_rate"] if "sample_rate" in data.keys() else None
    
    assert input_audio is not None, "Không có input của audio để inference"
    print("-"*80)
    print(input_audio)
    # run model
    model_outputs = cls_number.run(input_audio, sample_rate)

    response = {
        "labels": model_outputs    
    }

    return response