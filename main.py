import time
import tkinter as tk
import threading
import wave

import pyaudio

from event import get_fish_slices, send_audio
from PIL import Image, ImageTk


class FishWeightApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng dụng Cân Cá")
        self.root.geometry("800x500")

        # Tạo khung chính chia cửa sổ thành 2 phần
        self.main_frame = tk.Frame(root, bg="white")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Tạo khung bên trái
        self.left_frame = tk.Frame(self.main_frame, bg="#F0F0F0", bd=5, relief="groove")
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tạo khung bên phải
        self.right_frame = tk.Frame(self.main_frame, bg="#E0E0E0", bd=5, relief="groove")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Phần bên trái: Hiển thị số lát cá đã cân được từ cân điện tử
        self.fish_slices_label = tk.Label(self.left_frame, text="Số lát cá đã cân:", font=("Arial", 18, "bold"),
                                          bg="#F0F0F0")
        self.fish_slices_label.pack(pady=20)

        self.fish_slices = tk.StringVar(value=get_fish_slices())
        self.fish_slices_display = tk.Label(self.left_frame, textvariable=self.fish_slices, font=("Arial", 48, "bold"),
                                            bg="#F0F0F0", fg="#007BFF")

        # Tạo nút và đặt command đúng
        self.recognize = tk.Button(self.left_frame, text="Thu", font=("Arial", 18, "bold"), command=self.click)
        self.recognize.pack(pady=20)

        self.recording = False

        # Phần bên phải: Hiển thị ID khi người dùng đọc
        self.id_label = tk.Label(self.right_frame, text="ID người cân:", font=("Arial", 18, "bold"), bg="#E0E0E0")
        self.id_label.pack(pady=20)

        self.fish_slices_display.pack(pady=20)
        self.update_fish_slices()

        self.current_id = tk.StringVar()
        self.id_display = tk.Label(self.right_frame, textvariable=self.current_id, font=("Arial", 28, "bold"),
                                   bg="#E0E0E0", fg="#FF5733")
        self.id_display.pack(pady=20)

        # Thêm ảnh theo ID
        self.photo_label = tk.Label(self.right_frame, bg="#E0E0E0")
        self.photo_label.pack(pady=20)
        # Hiển thị hình ảnh khi khởi tạo
        self.display_image("./images/default.jpg")

    # Hàm click để thu âm
    def click(self):
        if self.recording:
            self.recording = False
            self.recognize.config(text="Thu")

        else:
            self.recording = True
            self.recognize.config(text="Dừng")
            threading.Thread(target=self.update_id).start()

    # Hàm cập nhật ID, hình ảnh
    def update_id(self):
        id_user = self.get_id()
        self.current_id.set(id_user)
        self.display_image(id_user)

    # Hàm lấy id sau khi đưa qua model
    def get_id(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
        frames = []

        while self.recording:
            data = stream.read(1024)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        audio.terminate()

        file_path_save = f"upload/record{int(time.time())}.wav"

        print(f"File Path: {file_path_save}")

        sound_file = wave.open(file_path_save, "wb")
        sound_file.setnchannels(1)
        sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        sound_file.setframerate(44100)
        sound_file.writeframes(b"".join(frames))
        sound_file.close()

        numbers = send_audio(frames)
        # print(f"Result: {numbers}")

        return numbers

    # Hàm cập nhật số cân sau mỗi 3s
    def update_fish_slices(self):
        self.fish_slices.set(get_fish_slices())
        self.root.after(3000, self.update_fish_slices)

    # Hàm hiển thị hình ảnh
    def display_image(self, id_user):
        try:
            image_path = f"./images/{id_user}.jpg"  # Đường dẫn đến ảnh tương ứng với ID
            image = Image.open(image_path)
        except FileNotFoundError:
            print(f"Không tìm thấy ảnh")
            # Nếu không tìm thấy ảnh, sử dụng ảnh mặc định hoặc không hiển thị ảnh
            image_path = "./images/default.jpg"  # Đường dẫn đến ảnh mặc định
            image = Image.open(image_path)

        image = image.resize((200, 200), Image.LANCZOS)  # Thay đổi kích thước ảnh nếu cần
        photo = ImageTk.PhotoImage(image)
        self.photo_label.config(image=photo)
        self.photo_label.image = photo  # Giữ tham chiếu để tránh bị thu hồi bởi Python GC


if __name__ == "__main__":
    root = tk.Tk()
    app = FishWeightApp(root)
    root.mainloop()
