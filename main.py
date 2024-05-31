import os
import random
import yaml
import click

@click.group()
def root():
    pass

@root.command("train")
@click.option("--config", default="default.yaml", help="Config path")
@click.option("--outfile", default="log.log", help="Config path")
@click.option("--is_aug", default=True, help="Config path")
def train(config, outfile, is_aug):
    from torch.utils.data import ConcatDataset
    from speech_number.dataset.dataset_cls import WhisperClsDataset
    from speech_number.model import WhisperEncoderCustomize
    from transformers import WhisperFeatureExtractor
    from speech_number.trainer import Trainer
    
    # load config
    config_path = "./configs/" + config
    outfile = "./log_output/" + outfile
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    outfile = open(outfile, "w")

    def train_(best_score, seed, is_aug):
        # save config
        os.makedirs(checkpoint_dir, exist_ok=True)

        model = WhisperEncoderCustomize.from_pretrained(config["train"]["pre_trained_model"])
        feature_extractor = WhisperFeatureExtractor.from_pretrained(config["train"]["pre_trained_model"])

        # load data
        if not is_aug:
            data_train = WhisperClsDataset(
                data_audio_dir=config["data"]["train_data_dir"],
                data_csv_dir=config["data"]["train_data_csv"],
                vocab_dir=config["data"]["vocab_dir"],
                feature_extractor=feature_extractor
            )
        else:
            data_train = WhisperClsDataset(
                data_audio_dir=config["data"]["train_data_dir"],
                data_csv_dir=config["data"]["train_data_csv"],
                vocab_dir=config["data"]["vocab_dir"],
                feature_extractor=feature_extractor
            )
            
            data_aug = WhisperClsDataset(
                data_audio_dir=config["data"]["train_aug_dir"],
                data_csv_dir=config["data"]["train_aug_csv"],
                vocab_dir=config["data"]["vocab_dir"],
                feature_extractor=feature_extractor
            )
            
            data_train = ConcatDataset([data_train, data_aug])
        
        data_test = WhisperClsDataset(
            data_audio_dir=config["data"]["test_data_dir"],
            data_csv_dir=config["data"]["test_data_csv"],
            vocab_dir=config["data"]["vocab_dir"],
            feature_extractor=feature_extractor
        )

        data_tests = {
            "train": data_train,
            "test": data_test
        }

        # setup trainer
        trainer = Trainer(
            epoch_num=config["train"]["epoch_num"],
            train_batch_size=config["train"]["train_batch_size"],
            test_batch_size=config["train"]["test_batch_size"],
            lr=config["train"]["lr"],
            outfile=outfile
        )
        trainer.setup(
            model=model,
            data_train=data_train,
            data_tests=data_tests,
            seed=seed
        )

        # start training
        best_score = trainer.train(
            seed = seed,
            checkpoint_dir=checkpoint_dir,
            best_checkpoint=config["train"]["best_checkpoint"],
            early_stopping=config["train"]["early_stopping"],
            best_score=best_score
        )

        return best_score

    best_score = config["train"]["best_score"]
    checkpoint_dir = config["train"]["checkpoint_dir"]
    n_train = config["train"]["n_train"]
    seeds = [config["train"]["seed_best"]] + [random.randint(0, 1000) for _ in range(n_train-1)]
    for train_time in range(n_train):
        score = train_(best_score, seeds[train_time], is_aug)
        if score > best_score:
            best_score = score
            seed = seeds[train_time]

    config["train"]["seed_best"] = seed
    config["train"]["best_score"] = best_score
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    print(f"Điểm tốt nhất {best_score} tại seed {seed}")


@root.command("serve")
@click.option("--config", default="default.yaml", help="Config path")
def serve(config):
    from speech_number.service.service_predict import start_aap_service
    
    config_path = "./configs/" + config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    start_aap_service(config["service"]["checkpoint_dir"])
    

@root.command("split_train_test")
@click.option("--config", default="default.yaml", help="Config path")
@click.option("--test_split", default=0.2, help="Config path")
def serve(config, test_split):
    from utils.split_train_test import split_train_test
    
    config_path = "./configs/" + config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    split_train_test(config, float(test_split))

@root.command("augment")
@click.option("--config", default="default.yaml", help="Config path")
def serve(config): 
    from speech_number.dataset.augment import AugmentData
    
    config_path = "./configs/" + config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    augment = AugmentData(input_dir=config["data"]["train_data_dir"], output_dir=config["data"]["train_aug_dir"])

    data_config = config["augment"] | config["data"]
    augment.run(config=data_config)
    
    
if __name__ == "__main__":
    root()