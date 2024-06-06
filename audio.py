import threading
import tkinter as tk
import os
import wave
import time

import pyaudio
import numpy as np
import requests
import io


class VoiceUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Voice Recorder")
        self.root.resizable(True, True)

        # Tạo nút và đặt command đúng
        self.button = tk.Button(self.root, text="Thu", font=("Arial", 120, "bold"), command=self.click)
        self.button.pack(pady=20)

        self.recording = False

        self.response_label = tk.Label(self.root, text="Kết quả", font=("Arial", 20, "bold"))
        self.response_label.pack(pady=20)

        self.root.mainloop()

    def click(self):
        if self.recording:
            self.recording = False
            self.button.config(text="Thu")
        else:
            self.recording = True
            self.button.config(text="Dừng")
            threading.Thread(target=self.record).start()

    def record(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

        frames = []
        while self.recording:
            data = stream.read(1024)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        audio.terminate()

        sound_file = wave.open(f"./upload/record{int(time.time())}", "wb")
        sound_file.setnchannels(1)
        sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        sound_file.setframerate(44100)
        sound_file.writeframes(b"".join(frames))
        sound_file.close()
        self.send_audio(frames)

    def send_audio(self, frames):
        text = "anh bảy"
        username = "Hello"
        country = "Hello"
        age = "Hello"

        audio_data = b''.join(frames)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        if len(audio_array) > 0:
            print("Có audio")
            # file_data = {"file": ("recorded_audio.wav", io.BytesIO(audio_data), "audio/wav")}

            # data_package = {"audio": "","text_target": text}
            #
            # response = requests.post("http://127.0.0.1:8000/danangvsr/vmd", data=data_package)

            # if response.status_code == 200:
            #     result = response.json()
            #     self.response_label.config(text=f"Result: {result}")
            # else:
            #     self.response_label.config(text=f"Failed to fetch data. Status code: {response.status_code}")
        else:
            self.response_label.config(text="The audio data is empty.")


# Chạy ứng dụng
VoiceUI()
