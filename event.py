import random
import numpy as np


def get_fish_slices():
    # Thay thế bằng mã thực tế để lấy dữ liệu từ cân điện tử
    return random.randint(0, 20)  # Giá trị mẫu


def send_audio(frames):
    audio_data = b''.join(frames)
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    if len(audio_array) > 0:
        # file_data = {"file": ("recorded_audio.wav", io.BytesIO(audio_data), "audio/wav")}

        # data_package = {"audio": "","text_target": text}
        #
        # response = requests.post("http://127.0.0.1:8000/danangvsr/vmd", data=data_package)

        # if response.status_code == 200:
        #     result = response.json()
        #     self.response_label.config(text=f"Result: {result}")
        # else:
        #     self.response_label.config(text=f"Failed to fetch data. Status code: {response.status_code}")
        print("Có audio")
        return "Có audio"
    else:
        return "Âm thanh lỗi!!"
