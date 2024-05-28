from fastapi import FastAPI
import uvicorn

from speech_number.service.service_class import InferenceRequest, InferenceResponse
from speech_number.service.service_predict import ClsNumber


app = FastAPI()
cls_number = ClsNumber("./exp/models/best")

@app.post("/cls_number/infer")
async def infer(data: InferenceRequest) -> InferenceResponse:
    
    input_audio = data.input_audio
    sample_rate = data.sample_rate # if sample_rate in data.keys() else None


    assert input_audio is not None, "Không có input của audio để inference"
    # run model
    model_outputs = cls_number.run(input_audio, sample_rate)

    response = {
        "label": model_outputs    
    }

    return response

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8088, reload=True)