import librosa
import soundfile as sf
import os
import random
import pandas as pd

class AugmentData:
    def __init__(self, input_dir, output_dir) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir

        self._remove_file_in_folder(output_dir)

    def _remove_file_in_folder(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        file_list = os.listdir(folder_path)
        for file_name in file_list:
            file_path = os.path.join(folder_path, file_name)
            os.remove(file_path)
            
    def change_speech(self, input_file: str, sample_rate: int, speed_factor: float, type: int):
        """
        0 < speed_factor < 1: tốc độ chậm
        speed_factor = 1: mặc định
        speech_factor > 1: tăng tốc độ
        """
        
        audio, _ = librosa.load(os.path.join(self.input_dir, input_file), sr=sample_rate)

        audio_changed = librosa.effects.time_stretch(audio, rate=speed_factor)

        output_path = os.path.join(self.output_dir, f"aug_speech_{type}_{input_file}")
        sf.write(output_path, audio_changed, sample_rate)

        return f"aug_speech_{type}_{input_file}"
    
    def change_pitch(self, input_file: str, sample_rate: int, n_steps: float, type: int):
        """
        n_steps < 0: giảm cao độ
        n_steps: 0 là mặc định
        n_setps > 0 tăng cao độ
        """
        audio, _ = librosa.load(os.path.join(self.input_dir, input_file), sr=sample_rate)
        
        audio_changed = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps)
        
        output_path = os.path.join(self.output_dir, f"aug_pitch_{type}_{input_file}")
        sf.write(output_path, audio_changed, sample_rate)
        return f"aug_pitch_{type}_{input_file}"
    
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

        output_path = os.path.join(self.output_dir, f"aug_volume_{type}_{input_file}")
        sf.write(output_path, gain_audio, sample_rate)
    
        return f"aug_volume_{type}_{input_file}"

    def run(self, config: dict = None):
        functions = ["change_speech", "change_pitch", "change_volume"]

        df = pd.read_csv(config["train_data_csv"])

        data_row = [] 
        for _, row in df.iterrows():
            
            loop = random.randint(1, len(functions))

            random_functions = random.sample(functions, loop)

            for func in random_functions:
                choose = random.randint(0, 1)
                method = getattr(self, func)
                file_name = method(row['path'], 16000, config[func][choose], choose)

                data_row.append({
                    "id": file_name[:-4],
                    "path": file_name,
                    "label": row["label"]
                })
            
        dataFrame = pd.DataFrame(data_row)

        dataFrame.to_csv(config["train_aug_csv"], mode="a", header=False, index=False)