import random
import speech_recognition as sr


def get_fish_slices():
    # Thay thế bằng mã thực tế để lấy dữ liệu từ cân điện tử
    return random.randint(0, 20)  # Giá trị mẫu


def get_id_user_from_voice():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=2)  # Lắng nghe tối đa 2 giây
        except sr.WaitTimeoutError:
            print("Hết thời gian lắng nghe.")
            return "Hết thời gian"

    try:
        id_user = recognizer.recognize_google(audio, language='vi-VN')
        print(f"ID người dùng: {id_user}")
        return id_user
    except sr.UnknownValueError:
        print("Không thể nhận diện.")
        return "Unknown"
    except sr.RequestError as e:
        print(f"Lỗi khi yêu cầu dịch vụ nhận diện giọng nói; {e}")
        return f"Lỗi khi yêu cầu dịch vụ nhận diện giọng nói; {e}"
