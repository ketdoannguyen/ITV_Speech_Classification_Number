import os
import shutil

import pandas as pd
import yaml


def _split_csv(raw_data_csv, test_split: float):
    df_all = pd.read_csv(raw_data_csv)
    print(len(df_all))
    label_values = list(set(df_all['label'].tolist()))
    num_samples_test = round(test_split * len(df_all))
    df_test = []
    count_test = 0
    for label in label_values:
        df_per_label = df_all.loc[df_all["label"] == label]
        sample_per_label = round(test_split * len(df_per_label))
        if sample_per_label>0:
            df_test.append(df_per_label.sample(n=sample_per_label, replace=False))
            count_test += sample_per_label
            
    df_test = pd.concat(df_test, ignore_index=False)
    idx_diff = df_all.index.difference(df_test.index)
    df_diff = df_all.loc[idx_diff]
    if num_samples_test - count_test > 0:
        samples_diff = df_diff.sample(n=num_samples_test - count_test, replace=False)
        df_test = pd.concat([df_test, samples_diff], axis=0, ignore_index=False)
    
    # Get train data
    idx_diff = df_all.index.difference(df_test.index)
    df_train = df_all.loc[idx_diff]
    
    return df_train, df_test
    
def split_train_test(config, test_split):
    df_train, df_test = _split_csv(config["data"]["raw_data_csv"], test_split=test_split)
    
    folder_raw = config["data"]["raw_data_dir"]
    folder_train = config["data"]["train_data_dir"]
    folder_test = config["data"]["test_data_dir"]
    
    _remove_file_in_folder(folder_train)
    _remove_file_in_folder(folder_test)
    
    _copy_file(df_train, folder_raw, folder_train)
    _copy_file(df_test, folder_raw, folder_test)
        
    df_train.to_csv(config["data"]["train_data_csv"], index=False)
    df_test.to_csv(config["data"]["test_data_csv"], index=False)
    
    print("Thành công")
    print(df_train['label'].value_counts())
    print(df_test['label'].value_counts())
    

def _remove_file_in_folder(folder_path):
    if os.path.exists(folder_path):
        file_list = os.listdir(folder_path)
        for file_name in file_list:
            file_path = os.path.join(folder_path, file_name)
            os.remove(file_path)


def _copy_file(df, folder_old, folder_new):
    if not os.path.exists(folder_new):
        os.makedirs(folder_new)
    path_audios = df['path'].tolist()
    for path in path_audios:
        shutil.copy(folder_old+path, folder_new)

if __name__ == "__main__":
    config_path = "./configs/default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    

    train, test = split_train_test(config, 0.2)
    
    train_c = train['label'].value_counts()
    test_c = test['label'].value_counts()
    
    print(train_c)
    
    print(test_c)
    
    