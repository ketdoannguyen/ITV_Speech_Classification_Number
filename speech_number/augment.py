import librosa
import soundfile as sf
import os
import random

class AugmentData:
    def __init__(self, input_dir, output_dir) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir

    def change_speech(self, input_file: str, sample_rate: int, speed_factor: float, type: int):
        """
        0 < speed_factor < 1: tốc độ chậm
        speed_factor = 1: mặc định
        speech_factor > 1: tăng tốc độ
        """
        audio, _ = librosa.load(os.path.join(self.input_dir, input_file), sr=sample_rate)

        audio_changed = librosa.effects.time_stretch(audio, rate=speed_factor)

        sf.write(os.path.join(self.output_dir, f"change_speech_{type}_{input_file}"), audio_changed, sample_rate)

    def change_pitch(self, input_file: str, sample_rate: int, n_steps: float, type: int):
        """
        n_steps < 0: giảm cao độ
        n_steps: 0 là mặc định
        n_setps > 0 tăng cao độ
        """
        audio, _ = librosa.load(os.path.join(self.input_dir, input_file), sr=sample_rate)
        
        audio_changed = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps)
        
        sf.write(os.path.join(self.output_dir, f"change_pitch_{type}_{input_file}"), audio_changed, sample_rate)

    
    def change_volume(self, input_file: str, sample_rate: int, gain: float, type: int):
        """
        gain dùng để điều chỉnh to nhỏ với:
            0: Là không nghe gì cả
            0 < gain < 1: âm sẽ nghe nhỏ
            gain = 1: Nghe mặc định
            gain > 1: âm sẽ to hơn
        """
        audio, _ = librosa.load(os.path.join(self.input_dir, input_file), sr=sample_rate)

        gain_audio = audio * gain

        sf.write(os.path.join(self.output_dir, f"change_volume_{type}_{input_file}"), gain_audio, sample_rate)

    def run(self, config: dict = None):
        functions = ["change_speech", "change_pitch", "change_volume"]
        filenames = os.listdir(self.input_dir)

        for file in filenames:
            loop = random.randint(1, len(functions))

            random_functions = random.sample(functions, loop)

            for func in random_functions:
                choose = random.randint(0, 1)
                method = getattr(self, func)
                method(file, 16000, config[func][choose], choose)