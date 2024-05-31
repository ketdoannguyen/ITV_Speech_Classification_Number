import streamlit as st
import numpy as np
import soundfile as sf
import time
from st_audiorec import st_audiorec
import os
import torchaudio
import requests
from supabase import create_client, Client

# init DB
url: str = "https://yyciwuqbkcqecbrqholh.supabase.co"
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inl5Y2l3dXFia2NxZWNicnFob2xoIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcxNjI5NTEwNSwiZXhwIjoyMDMxODcxMTA1fQ.5mRyn4e7g1PKBnh2N6g10ISkp7CvQnX2owbWQLe9lnQ"
DB: Client = create_client(supabase_url=url, supabase_key=key)

# Global variable
OUT_WAV_FILE = None


def colorize(value):
    if value == 1:
        return "color: green"
    elif value == 0:
        return "color: red"
    else:
        return ""


def handle_feedback(feedback, r, DB):
    if OUT_WAV_FILE:
        bucket_res = DB.storage.from_("data-feedback").upload(file=OUT_WAV_FILE,
                                                              path=f"{OUT_WAV_FILE}",
                                                              file_options={"content-type": "audio/wav"})
        if bucket_res:
            wav_url = DB.storage.from_("data-feedback").get_public_url(path=OUT_WAV_FILE)
            print(f"Wav url: {wav_url}")

            feedback_data = {
                "audio_url": wav_url,
                "label_predicted": r.json().get('label'),
                "feedback": feedback if feedback else None
            }

            response = DB.table("feedback-data").insert(feedback_data).execute()
            print(f"DB: {response}")

            if response:
                st.success("Cảm ơn bạn đã phản hồi!")
                if os.path.exists(OUT_WAV_FILE):
                    os.remove(OUT_WAV_FILE)
            else:
                st.error("Lỗi!!!")
        else:
            st.error("Không thể kết nối Bucket!!!")


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
                    sf.write(OUT_WAV_FILE, audio_array, 44100)
                    waveform, sample_rate = torchaudio.load(OUT_WAV_FILE)
                    os.remove(OUT_WAV_FILE)

                    # call api
                    # data to be sent to api
                    data = {
                        "input_audio": waveform.tolist(),
                        "sample_rate": sample_rate
                    }
                    # url="http://127.0.0.1:8000/cls_number/infer"
                    url = "https://cls-number-nkd.onrender.com/cls_number/infer"
                    # sending post request and saving response as response object
                    r = requests.post(url=url, json=data)

                    # Hiển thị thông báo
                    print(f"Path khi chưa gửi feedback: {OUT_WAV_FILE}")
                    print(r.json())
                    st.success(f"Kết quả dự đoán: {r.json()['label']}")
                else:
                    st.warning("The audio data is empty.")

        st.write("""
        ### Phản hồi của khách hàng
        """)
        feedback = st.text_input("Phản hồi", placeholder="Nhập phản hồi ở đây...")
        if st.button("Gửi phản hồi"):
            print(f"Path khi gửi feedback: {OUT_WAV_FILE}")
            handle_feedback(feedback, r, DB)


if __name__ == "__main__":
    st.set_page_config(page_title="Predict Numbers", layout="wide")

    main()
