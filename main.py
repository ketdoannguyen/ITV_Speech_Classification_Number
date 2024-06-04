import streamlit as st
import numpy as np
import soundfile as sf
import time

from multidict import istr
from st_audiorec import st_audiorec
import os
import torchaudio
import requests
from supabase import create_client, Client


def colorize(value):
    if value == 1:
        return "color: green"
    elif value == 0:
        return "color: red"
    else:
        return ""


def handle_feedback(feedback, r, OUT_WAV_FILE, is_click=False):
    # init DB
    url: str = "https://yyciwuqbkcqecbrqholh.supabase.co"
    key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inl5Y2l3dXFia2NxZWNicnFob2xoIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcxNjI5NTEwNSwiZXhwIjoyMDMxODcxMTA1fQ.5mRyn4e7g1PKBnh2N6g10ISkp7CvQnX2owbWQLe9lnQ"
    DB: Client = create_client(supabase_url=url, supabase_key=key)

    if not is_click:
        try:
            # Lưu trữ file wav vào bucket
            bucket_res = DB.storage.from_("data-feedback").upload(file=OUT_WAV_FILE, path=f"{OUT_WAV_FILE}",
                                                                  file_options={"content-type": "audio/wav"})
            print(f"Bucket: {bucket_res}")

            # Lấy file wav url
            wav_url = DB.storage.from_("data-feedback").get_public_url(path=OUT_WAV_FILE)
            wav_url = wav_url[:-1]  # Xóa ký tự cuối "?"

            print(f"Wav url for insert: {wav_url}")

            insert_data = {
                "audio_url": wav_url, "label_predicted": r.json().get('label'), "feedback": None
            }

            # Insert vào bảng
            response = DB.table("feedback-data").insert(insert_data).execute()
            print(f"DB: {response}")
        except:
            print("------BUG")
    else:
        update_feedback = {
            "feedback": feedback
        }

        wav_url = DB.storage.from_("data-feedback").get_public_url(path=OUT_WAV_FILE)
        wav_url = wav_url[:-1]  # Xóa ký tự cuối "?"

        print(f"Wav url for update: {wav_url}")

        # Update bảng theo file wav url
        response = DB.table("feedback-data").update(update_feedback).eq('audio_url', wav_url).execute()
        print(f"DB: {response}")

        if response:
            st.success("Cảm ơn bạn đã phản hồi!")
        else:
            st.error("Lỗi!!!")


def main():
    st.markdown("<h1>Nhận diện đếm số</h1>", unsafe_allow_html=True)

    # UI Audio
    wav_audio_data = st_audiorec()

    predict_button = st.button('Dự đoán')

    if not st.session_state.get('predict'):
        st.session_state['predict'] = predict_button

    # handle click button
    if st.session_state['predict'] and wav_audio_data:
        # Convert audio_bytes to a NumPy array
        audio_array = np.frombuffer(wav_audio_data, dtype=np.int32)

        if len(audio_array) > 0:
            # tạo file wav từ audio array
            OUT_WAV_FILE = f"./upload/record{int(time.time())}.wav"
            sf.write(OUT_WAV_FILE, audio_array, 44100)
            waveform, sample_rate = torchaudio.load(OUT_WAV_FILE)

            # call api
            # data to be sent to api
            data = {
                "input_audio": waveform.tolist(),
                "sample_rate": sample_rate
            }

            url = "http://127.0.0.1:8000/cls_number/infer"
            r = requests.post(url=url, json=data)

            # Hiển thị thông báo
            st.success(f"Kết quả dự đoán: {r.json()['label']}")
            print(r.json())
            handle_feedback(None, r, OUT_WAV_FILE, is_click=False)
            if os.path.exists(OUT_WAV_FILE):
                os.remove(OUT_WAV_FILE)
        else:
            st.warning("Bản ghi bị lỗi hãy thử lại!")

        st.write("""
        ### Phản hồi của khách hàng
        """)
        feedback = st.text_input("Phản hồi", placeholder="Nhập phản hồi ở đây...")
        if st.button("Gửi phản hồi"):
            print(f"Path khi gửi feedback: {OUT_WAV_FILE}")
            handle_feedback(feedback, r, OUT_WAV_FILE, is_click=True)
            st.session_state['predict'] = False

    else:
        st.warning("Vui lòng ghi âm trước khi dự đoán!")


if __name__ == "__main__":
    st.set_page_config(page_title="Predict Numbers")

    main()
