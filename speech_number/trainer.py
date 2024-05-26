import wandb
import torch
import os
from transformers import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, epoch_num, train_batch_size, test_batch_size, lr, outfile):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.epoch_num = epoch_num
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.lr = lr
        self.outfile = outfile

    def setup(self, model, data_train, data_tests, seed):
        torch.manual_seed(seed)
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # setup train object
        self.optimizer = AdamW(
            params=self.model.parameters(),
            lr=self.lr,
            correct_bias=True,
            weight_decay=1e-5,
        )
        self.scheduler = StepLR(self.optimizer, step_size=2, gamma=0.7)

        # set data info
        self.data_train = data_train
        self.data_tests = data_tests

        # setup dataloader
        self.dataloader_train = DataLoader(
            data_train,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=10
        )

        self.dataloader_tests = {
            k: DataLoader(
                data_tests[k],
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=10
            )
            for k in data_tests.keys()
        }

    def train(
        self,
        seed=None,
        early_stopping=None,
        best_checkpoint=None,
        best_score=None,
        checkpoint_dir=None
    ):
        wandb.init(project="[ITV] Count Number", name=f"Test version 1 seed {seed}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # best checkpoint params
        assert best_checkpoint is not None, "'best_checkpoint' must be not None"

        if early_stopping is not None:
            last_loss = 999999
            trigger_times = 0

        for epoch in range(1, self.epoch_num + 1):
            # pre epoch
            self.model.pre_epoch(self, epoch)

            # train epoch
            train_log_info = self.model.train_epoch(self, epoch, self.outfile)
            print(train_log_info)

            # update lr
            self.scheduler.step()

            # test
            test_log_info = {}
            for data_type in self.dataloader_tests.keys():
                test_log_info[data_type] = self.test(
                    self.dataloader_tests[data_type], 
                    epoch=epoch, 
                    data_name= f"{data_type}", 
                    outfile=self.outfile
                )


            # best check point
            current_score = test_log_info[best_checkpoint["data_name"]][best_checkpoint["metric_type"]]
            if current_score > best_score:
                best_score = current_score
                self.model.save_pretrained(checkpoint_dir)

            # early stopping
            if early_stopping is not None:
                current_loss = test_log_info[early_stopping["data_name"]][
                    early_stopping["loss_type"]
                ]

                if current_loss > last_loss:
                    trigger_times += 1

                last_loss = current_loss

                if trigger_times >= early_stopping["patience"]:
                    print("Early stopping!")
                    break
        wandb.finish()
        return best_score

    def test(self, dataloader, epoch, data_name, outfile):
        log_info = self.model.test_epoch(
            self, 
            dataloader=dataloader, 
            epoch=epoch, 
            data_name=data_name, 
            outfile=outfile
        )

        wandb_log = {}
        for log_key in log_info.keys():
            wandb_log[f"{data_name}/{log_key}"] = log_info[log_key]
        wandb_log[f"{data_name}/epoch"] = epoch
        wandb.log(wandb_log)
        return log_info
