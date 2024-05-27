from typing import Union, List
from pydantic import BaseModel
import torch


class InferenceResponse(BaseModel):
    labels: str


class InferenceRequest(BaseModel):
    input_audio: Union[str, torch.Tensor] = None
    sample_rate: int = None

    class Config:
        arbitrary_types_allowed = True
