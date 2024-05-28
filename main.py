import streamlit as st
import numpy as np
import soundfile as sf
import time
from st_audiorec import st_audiorec
import os
import torchaudio
import requests



def colorize(value):
    if value == 1:
        return "color: green"
    elif value == 0:
        return "color: red"
    else:
        return ""


def main():
    _, cl1, _ = st.columns([4, 5, 4])
    # UI
    with cl1:
        st.markdown("<h1>Nhận diện đếm số</h1>", unsafe_allow_html=True)

        # UI Audio
        wav_audio_data = st_audiorec()
        
        # handle click button
        if st.button("Dự đoán") and wav_audio_data:
            with st.spinner('Đợi trong giây lát...'):
                # Convert audio_bytes to a NumPy array
                audio_array = np.frombuffer(wav_audio_data, dtype=np.int32)
                
                if len(audio_array) > 0:
                    # tạo file wav từ audio array
                    OUT_WAV_FILE = f"{int(time.time())}.wav"
                    sf.write(OUT_WAV_FILE, audio_array , 44100)
                    waveform, sample_rate = torchaudio.load(OUT_WAV_FILE)
                    os.remove(OUT_WAV_FILE)
                    
                    #call api
                    # data to be sent to api
                    data = {
                        "input_audio": waveform.tolist(),
                        "sample_rate": sample_rate
                    } 
                    # url="http://127.0.0.1:8000/cls_number/infer"
                    url="https://cls-number-nkd.onrender.com/cls_number/infer"
                    # sending post request and saving response as response object
                    r = requests.post(url=url, json=data)
                    
                    # Hiển thị thông báo
                    st.success(f"kết quả dự đoán: {r.json()['label']}")
                else:
                    st.warning("The audio data is empty.")


if __name__ == "__main__":
    st.set_page_config(page_title="Collect Data", layout="wide")
    main()