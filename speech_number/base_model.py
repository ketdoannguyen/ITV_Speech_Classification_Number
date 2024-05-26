from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def pre_epoch(self, trainer, epoch):
        pass

    @abstractmethod
    def train_epoch(self, trainer, epoch, outfile):
        pass

    @abstractmethod
    def test_epoch(self, trainer, dataloader, epoch, data_name, outfile):
        pass