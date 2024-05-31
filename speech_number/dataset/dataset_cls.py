import json
import torch
import torchaudio
import tqdm
from transformers import AutoFeatureExtractor
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


class WhisperClsDataset(Dataset):
    def __init__(self, data_audio_dir, data_csv_dir, vocab_dir, feature_extractor):
        # init
        self.feature_extractor = feature_extractor
        self.data_audio_dir = data_audio_dir
        self.data_count_number = self.get_data(data_csv_dir)
        self.vocab_dir = vocab_dir
        self.label2index, self.index2label, self.n_class = self._get_info_label()
        self.data_count_number = self.data_count_number.map(self._process_label)
        self.data_count_number.set_format(type='torch', columns=['label', 'input_features'])

    def get_data(self, data_csv_dir):
        data_count_number = load_dataset("csv", data_files=data_csv_dir)
        data_count_number = data_count_number.map(self._process_audio)

        return data_count_number


    def _get_info_label(self):
        with open(self.vocab_dir) as f:
            vocab = json.load(f)
            
        label2index = vocab["label2index"]
        index2label = vocab["index2label"]
        
        return label2index, index2label, len(label2index)

    def _process_audio(self, batch):
        new_sample_rate = 16000
        waveform, sample_rate = torchaudio.load(self.data_audio_dir + str(batch["path"]))

        waveform = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform)
        waveform = waveform.squeeze()
        batch["input_features"] = self.feature_extractor(waveform, sampling_rate=new_sample_rate, return_tensors="pt").input_features.squeeze(0)
        return batch

    def _process_label(self, batch):
        batch["label"] = self.label2index.get(batch['label'], -1)
        return batch

    def __len__(self):
        return len(self.data_count_number['train'])

    def __getitem__(self, index):
        data_index = self.data_count_number['train'][index]

        return {
            "input_features": data_index["input_features"],
            "labels": data_index["label"]
        }
    
    def custom_collate_fn(self, batch):
        return torch.stack(batch, dim=0)


if __name__ == "__main__":
    model_name = "vinai/PhoWhisper-tiny"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    dataset_whisper = WhisperClsDataset("./dataset/test_data/audio/", "./dataset/test_data/csv_data.csv", "./dataset/vocab_dataset.json", feature_extractor)

    print(dataset_whisper.data_count_number)
    print(dataset_whisper.data_count_number["train"][[0]])
    print(dataset_whisper.label2index)

    dataloader_train = DataLoader(
        dataset_whisper,
        batch_size=16,
        shuffle=True,
        num_workers=10
    )
    for batch_data in tqdm.tqdm(dataloader_train, ncols=100):
            input_features = batch_data["input_features"]
            print(input_features.shape)
            labels = batch_data["labels"]
            print(labels.shape)
