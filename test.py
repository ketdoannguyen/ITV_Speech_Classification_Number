
from speech_number.service.service_predict import infer

data = {
    "input_audio": "/media/vkuai/01D7F711799324B0/ketdoan/ITV_Speech_Classification_Number/dataset/audio_count_web/xac_nhan_1.wav",
}
print(infer("./exp/models/best", data))