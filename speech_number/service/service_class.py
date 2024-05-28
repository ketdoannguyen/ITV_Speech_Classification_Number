from typing import Union
from pydantic import BaseModel

class InferenceResponse(BaseModel):
    label: str

class InferenceRequest(BaseModel):
    input_audio: Union[str, list, None] = None
    sample_rate: int = None

    class Config:
        arbitrary_types_allowed = True