import pandas as pd
import unicodedata
import os


def get_path_file_wav(data):
    return data.split("/")[-1]


def remove_tone_marks(text):
    text = text.replace("đ", "d")
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')


id_audio = {}


def process_id(text):
    global id_audio
    text = str(text)
    name_save_audio = remove_tone_marks(text.strip()).replace(" ", "_").replace("!", "")

    if name_save_audio not in id_audio.keys():
        index = 1
        id_audio[name_save_audio] = index
        name_save_audio += "_" + str(index)
    else:
        index = id_audio[name_save_audio] + 1
        id_audio[name_save_audio] = index
        name_save_audio += "_" + str(index)
    return name_save_audio


def convert_label(text):
    # dict_label = {'một': 1, 'hai': 2, 'ba': 3, 'bốn': 4, 'năm': 5,
    #               'sáu': 6, 'bảy': 7, 'tám': 8, 'chín': 9, 'mười': 10,
    #               'mười một': 11, 'mười hai': 12, 'mười ba': 13, 'mười bốn': 14, 'mười lăm': 15,
    #               'mười sáu': 16, 'mười bảy': 17, 'mười tám': 18, 'mười chín': 19, 'hai mươi': 20,
    #               'xác nhận': 'xác nhận', 'ok': 'xác nhận', 'làm lại': 'làm lại'}

    # for num_label in dict_label:
    #     if text == num_label:
    #         if isinstance(text, int):
    #             return int(dict_label[text])
    #         else:
    #             return str(dict_label[text])

    dict_label = {
        '1': 'một', '2': 'hai', '3': 'ba', '4': 'bốn', '5': 'năm', '6': 'sáu', '7': 'bảy', '8': 'tám', '9': 'chín',
        '10': 'mười', '11': 'mười một', '12': 'mười hai', '13': 'mười ba', '14': 'mười bốn', '15': 'mười lăm',
        '16': 'mười sáu', '17': 'mười bảy', '18': 'mười tám', '19': 'mười chín', '20': 'hai mươi',
        'xác nhận': 'xác nhận', 'Xác nhận': 'xác nhận', 'Làm lại': 'làm lại', 'làm lại': 'làm lại'
    }

    for label in dict_label:
        if text == label:
            return dict_label[label]


def convert_name_file(file_data, folder_path):
    data = pd.read_csv(file_data)
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            for url in data["audio_url"]:
                if url == filename:
                    old_name = data[data["audio_url"] == url]["audio_url"].values
                    old_name = ''.join(old_name)
                    name_new = data[data["audio_url"] == url]["id"].values
                    name_new = ''.join(name_new) + ".wav"
                    new_filename = filename.replace(old_name, name_new)
                    os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))


if __name__ == "__main__":
    file_save_csv_data_web = "./data_csv/"
    folder_path = './data_test/'

    data = pd.read_csv(file_save_csv_data_web)

    data["label"] = data["label"].map(convert_label)
    data["id"] = data["label"].map(process_id)
    data["path"] = data["id"].map(lambda x: "audio_count_test/" + x + ".wav")
    data["audio_url"] = data["audio_url"].map(get_path_file_wav)

    data = data.reindex(columns=["id", "label", "path", "audio_url"])
    data = data.sort_values(["label", "id"])
    data.to_csv(file_save_csv_data_web, index=False)

    convert_name_file(file_save_csv_data_web, folder_path)

    data = data.drop(columns=['audio_url'])
    data.to_csv(file_save_csv_data_web, index=False)
