import random
import time

import numpy as np
import requests
import torchaudio


def get_fish_slices():
    # Thay thế bằng mã thực tế để lấy dữ liệu từ cân điện tử
    return random.randint(0, 20)  # Giá trị mẫu


def send_audio(frame, OUT_WAV_FILE):
    audio_data = b''.join(frame)
    audio_array = np.frombuffer(audio_data, dtype=np.int32)

    if len(audio_array) > 0:
        print("Tiếp nhận Audio...")
        # tạo file wav từ audio array
        waveform, sample_rate = torchaudio.load(OUT_WAV_FILE)

        # Dữ liệu gửi đến API
        data = {
            "input_audio": waveform.tolist(),
            "sample_rate": sample_rate
        }

        time_start = time.time()
        # Địa chỉ gọi API
        url = "http://127.0.0.1:8000/cls_number/infer"
        r = requests.post(url=url, json=data)

        print(f"Chạy trong {time.time() - time_start}")

        return r.json()['label']

    else:
        return "Âm thanh lỗi!!"
