import streamlit as st
import numpy as np
import soundfile as sf
import time
from st_audiorec import st_audiorec
import os


def colorize(value):
    if value == 1:
        return "color: green"
    elif value == 0:
        return "color: red"
    else:
        return ""


def main():
    
    # UI
    st.markdown("<h1>Nhận diện đếm số</h1>", unsafe_allow_html=True)
    label = st.text_input("Label")
    st.markdown(f"""<div style="display: flex; gap: 10px"><p style='font-size: 15px; color: 'black'>Label: 
                                <span style='font-size: 20px; color: 'red'><strong>{label}</strong></span></p></div>""",
        unsafe_allow_html=True)

    # UI Audio
    wav_audio_data = st_audiorec()

    
    
    # handle click button
    if st.button("Lưu dữ liệu") and wav_audio_data:
        with st.spinner('Đợi trong giây lát...'):
            if label != '':
                # Convert audio_bytes to a NumPy array
                audio_array = np.frombuffer(wav_audio_data, dtype=np.int32)
                
                if len(audio_array) > 0:
                    # Thư mục chứa file audio
                    os.makedirs("./upload", exist_ok=True)
                    # tạo file wav từ audio array
                    OUT_WAV_FILE = f"./upload/{int(time.time())}_{label}.wav"
                    sf.write(OUT_WAV_FILE, audio_array , 44100)
                    
                    
                    # class api detect number bên đoàn
                    
                    
                    # Hiển thị thông báo
                    st.success(f"Audio data saved as {OUT_WAV_FILE}")
                else:
                    st.warning("The audio data is empty.")
            else:
                st.warning("Điền đầy đủ thông tin bạn nhé")


if __name__ == "__main__":
    st.set_page_config(page_title="Collect Data", layout="wide")
    main()